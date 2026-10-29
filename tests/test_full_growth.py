"""Main-rule growth integration; fixture XP is never a performance result."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import unittest
import numpy as np
import torch
from unittest.mock import patch
from pathlib import Path
import tempfile

from space_z.config import ObsConfig
from space_z.config.progression_definitions import SHIPS,STAT_UPGRADES,STAT_MAX_LEVEL
from space_z.rl.actions.discrete_action_decoder import DiscreteActionDecoder
from space_z.rl.training_wooki.environment import TrainingEnv


class FullGrowthTests(unittest.TestCase):
    def env(self):
        from space_z.rl.training_wooki import environment as module
        original=module.game_config
        def fixture(stage):
            cfg=original(stage);cfg.num_enemies=cfg.num_shapes=0;cfg.terrain.enabled=False
            return cfg
        with patch.object(module,'game_config',side_effect=fixture):
            env=TrainingEnv(max_steps=64,curriculum=False,full_growth=True)
        env.reset(seed=71)
        self.addCleanup(env.close)
        return env

    def award_to(self,env,level):
        # Explicit mechanics fixture, not a natural-play evaluation.
        p=env.env.game.progression
        while p.level<level:
            p._add_score(p.xp_to_next_level-p.xp)

    def test_factorized_choices_match_all_main_decoder_actions(self):
        from space_z.rl.training_wooki.growth import encode_growth_action
        decoder=DiscreteActionDecoder(enable_stat_upgrades=True,enable_evolution=True)
        combat_decoder=DiscreteActionDecoder()
        seen=set()
        for combat in range(54):
            for stat in range(6):
                for evolution in range(5):
                    value=encode_growth_action((combat,stat,evolution))
                    self.assertTrue(decoder.action_space.contains(value))
                    actual=decoder.decode(value);expected=combat_decoder.decode(combat)
                    self.assertEqual((actual.move_x,actual.move_y,actual.rotation,actual.want_fire),
                                     (expected.move_x,expected.move_y,expected.rotation,expected.want_fire))
                    self.assertEqual(actual.stat_upgrade_key,STAT_UPGRADES[stat-1].key if stat else None)
                    self.assertEqual(actual.evolution_choice,evolution-1 if evolution else None)
                    seen.add(value)
        self.assertEqual(len(seen),1620)

    def test_no_points_or_rewards_only_permits_no_growth(self):
        from space_z.rl.training_wooki.growth import legal_growth_masks
        env=self.env();obs=env._observation(env.builder.build(env.env.game))
        stat,evo=legal_growth_masks(torch.tensor(obs['vector'][None]))
        self.assertEqual(stat[0].tolist(),[True,False,False,False,False,False])
        self.assertEqual(evo[0].tolist(),[True,False,False,False,False])

    def test_stat_action_spends_one_point_and_changes_main_stats(self):
        env=self.env();p=env.env.game.progression
        p._add_score(p.xp_to_next_level)
        self.assertEqual(p.upgrade_points,1)
        hp=env.env.game.player.max_hp
        env.step(np.array([0,1,0]))
        self.assertEqual(p.upgrade_points,0)
        self.assertEqual(p.stat_levels['health'],1)
        self.assertGreater(env.env.game.player.max_hp,hp)

    def test_every_stat_caps_and_is_masked_after_maximum(self):
        from space_z.rl.training_wooki.growth import legal_growth_masks
        for index,card in enumerate(STAT_UPGRADES,1):
            env=self.env();p=env.env.game.progression
            p.upgrade_points=STAT_MAX_LEVEL+1
            for _ in range(STAT_MAX_LEVEL):
                env.step(np.array([0,index,0]))
            self.assertEqual(p.stat_levels[card.key],STAT_MAX_LEVEL)
            before=p.upgrade_points
            env.step(np.array([0,index,0]))
            self.assertEqual(p.upgrade_points,before)
            obs=env.builder.build(env.env.game)[:ObsConfig().vector_size]
            stat,_=legal_growth_masks(torch.tensor(obs[None]))
            self.assertFalse(stat[0,index])

    def test_all_ship_branches_and_mastery_apply_through_main_actions(self):
        reached=set()
        leaves=[ship for ship in SHIPS.values() if ship.level==30]
        for leaf in leaves:
            env=self.env();p=env.env.game.progression
            parent=SHIPS[leaf.parent];family=SHIPS[parent.parent]
            for level,key in ((10,family.key),(20,parent.key),(30,leaf.key),(40,leaf.key)):
                self.award_to(env,level)
                choices=p.reward_choices
                index=next(i for i,x in enumerate(choices) if x.key==key)
                before=p.special_levels[key]
                env.step(np.array([0,0,index+1]))
                self.assertEqual(env.env.game.player.job,key)
                self.assertEqual(p.special_levels[key],before+1)
                self.assertFalse(p.special_reward_levels)
                reached.add(key)
            env.close()
        expected={key for key,ship in SHIPS.items() if ship.level in (10,20,30)}
        self.assertEqual(reached,expected)

    def test_policy_never_selects_unavailable_growth_and_uses_available_points(self):
        from space_z.rl.training_wooki.runtime import make_vec_env,make_model
        from space_z.rl.training_wooki.settings import TrainSettings
        cfg=TrainSettings(full_growth=True,workers=1,curriculum=False,torch_threads=1)
        env=make_vec_env(cfg);self.addCleanup(env.close)
        model=make_model(cfg,env);obs=env.reset()
        for _ in range(8):
            action,_=model.predict(obs,deterministic=False)
            self.assertEqual(action.shape,(1,3));self.assertEqual(action[0,1:].tolist(),[0,0])
        raw=env.venv.venv.envs[0].env
        raw.env.game.progression._add_score(raw.env.game.progression.xp_to_next_level)
        # Obtain the new observation by a real no-op main action, not altered input flags.
        obs,_,_,_=env.step(np.array([[0,0,0]]))
        action,_=model.predict(obs,deterministic=True)
        self.assertIn(int(action[0,1]),range(1,6))
        env.step(action)
        self.assertEqual(raw.stat_upgrades,1)

    def test_available_evolution_masks_follow_current_main_choices(self):
        from space_z.rl.training_wooki.growth import legal_growth_masks
        env=self.env();self.award_to(env,10)
        obs=env.builder.build(env.env.game)[:ObsConfig().vector_size]
        _,evolution=legal_growth_masks(torch.tensor(obs[None]))
        self.assertFalse(evolution[0,0])
        self.assertEqual(int(evolution.sum()),len(env.env.game.progression.reward_choices))

    def test_policy_selects_real_evolution_and_main_promotes_enemies(self):
        from space_z.rl.training_wooki.runtime import make_vec_env,make_model
        from space_z.rl.training_wooki.settings import TrainSettings
        env=make_vec_env(TrainSettings(full_growth=True,workers=1,curriculum=False,torch_threads=1))
        self.addCleanup(env.close);model=make_model(TrainSettings(full_growth=True,workers=1,torch_threads=1),env)
        env.reset();raw=env.venv.venv.envs[0].env;p=raw.env.game.progression
        self.award_to(raw,10)
        obs,_,_,_=env.step(np.array([[0,0,0]]))
        action,_=model.predict(obs,deterministic=True)
        self.assertGreater(action[0,2],0)
        tensors,_=model.policy.obs_to_tensor(obs)
        distribution=model.policy.get_distribution(tensors)
        selected=distribution.get_actions(deterministic=True)
        model.policy.optimizer.zero_grad()
        (-distribution.log_prob(selected).mean()).backward()
        gradient=model.policy.action_net.weight.grad
        self.assertGreater(float(gradient[54:60].abs().sum()),0.)
        self.assertGreater(float(gradient[60:].abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.policy.combat.parameters()))
        env.step(action)
        self.assertEqual(raw.evolutions,1)
        self.assertEqual(SHIPS[raw.env.game.player.job].level,10)
        self.assertTrue(all(SHIPS[enemy.job].level==10 for enemy in raw.env.game.enemies))

    def test_growth_ppo_updates_saves_and_resumes(self):
        from space_z.rl.training_wooki.settings import TrainSettings
        from space_z.rl.training_wooki.train import run_training
        from space_z.rl.training_wooki.runtime import read_json,load_checkpoint
        cfg=TrainSettings(full_growth=True,corner_recovery=True,curriculum=False,
                          objective='score100',workers=1,total_steps=16,rollout_steps=8,
                          batch_size=8,epochs=1,max_steps=4,eval_steps=2,eval_episodes=1)
        with tempfile.TemporaryDirectory() as tmp:
            run_training(tmp,cfg,subprocess=False)
            state=read_json(Path(tmp)/'status.json')
            self.assertEqual(state['status'],'completed');self.assertGreater(state['updates'],0)
            m,e=load_checkpoint(tmp,read_json(Path(tmp)/'latest.json'))
            try:
                obs=e.reset();action,_=m.predict(obs)
                self.assertEqual(action.shape,(1,3))
                self.assertGreater(len(m.policy.optimizer.state),0)
                e.step(action)
            finally:
                e.close()
            cfg.total_steps=24;run_training(tmp,cfg,resume=True,subprocess=False)
            self.assertGreaterEqual(read_json(Path(tmp)/'status.json')['step'],24)


if __name__=='__main__':
    unittest.main()
