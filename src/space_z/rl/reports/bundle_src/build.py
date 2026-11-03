"""4개 계열의 계보 페이지를 읽어 전달용 리포트 번들(reports/bundle)을 새 디자인으로 생성한다."""
import html
import re
import shutil
import statistics
from pathlib import Path

from bs4 import BeautifulSoup, Tag

SRC = Path(__file__).resolve().parent
REPORTS = SRC.parent
RL = REPORTS.parent
OUT = REPORTS / 'bundle'
ZIP_BASE = REPORTS / 'space_z_rl_report_bundle'

STATUS_CLASSES = ('base', 'good', 'warn', 'bad', 'blend')
LEGEND = [('base', '기준점'), ('good', '개선·채택'), ('warn', '개선 없음·보류'), ('bad', '실패·폐기'), ('blend', '가중치 혼합')]

APPROACHES = [
    dict(
        key='vector_only', short='교차어텐션', source='gpt_training_vector_only/reports/lineage_prototype.html',
        name='벡터 교차어텐션 하이브리드', tag='교차어텐션 · 회전만 연속', tone='#4f46e5',
        summary='적·천체·탄환을 개체 토큰으로, 지형 격자를 CNN 토큰으로 인코딩한 뒤 교차어텐션으로 하나의 64차원 특징으로 묶습니다. '
                '행동은 회전만 연속(tanh-Gaussian)이고 이동·발사·강화·진화는 이산인 22개 파라미터 결합 분포에서 뽑습니다.',
        obs='벡터 관찰(개체·성장·지형), 이미지 없음',
        act='이동 9 · 회전 1(연속) · 발사 1 · 강화 6 · 진화 5 = 22',
        policy='개체 인코더 + 지형 CNN + 교차어텐션 → 정책/가치 MLP 128',
        result='004 기준점에서 <strong>보상 계수만 올린 007</strong>이 held-out 6시드 평균 107.5 → 171.7점(<strong>+59.7%</strong>). '
               '보상·학습률·환경 수를 함께 바꾼 008은 007을 넘지 못했습니다.',
        headline='007 · held-out +59.7%',
        extra_edges=[],
        arch=dict(cols=5, boxes=[
            ('obs', 0, 70, '벡터 관찰', '개체·성장·지형', 'io'),
            ('ent', 1, 10, '개체 인코더', '적·천체·탄환 토큰', 'core'),
            ('ter', 1, 130, '지형 CNN', '격자 → 40 토큰', 'core'),
            ('att', 2, 70, '교차어텐션', '4 head · 64차원', 'core'),
            ('pi', 3, 10, '정책 MLP', '128 · GELU', 'core'),
            ('vf', 3, 130, '가치 MLP', '128 · GELU', 'core'),
            ('act', 4, 10, 'Actor', '22 파라미터', 'head'),
            ('cri', 4, 130, 'Critic', '가치 1', 'head'),
        ], edges=[('obs', 'ent'), ('obs', 'ter'), ('ent', 'att'), ('ter', 'att'), ('att', 'pi'), ('att', 'vf'),
                  ('pi', 'act'), ('vf', 'cri')]),
    ),
    dict(
        key='rotation_continuous', short='회전 연속', source='gpt_training_rotation_continuous/reports/lineage_prototype.html',
        name='회전 연속 하이브리드', tag='독립 조준망 · 회전만 연속', tone='#0e9f6e',
        summary='원본 5,139차원 관찰을 개체·지형 인코더(256)로 압축하는 장면망이 이동·강화·진화를 고르고, 가까운 표적의 상대 좌표만 보는 '
                '독립 조준망이 회전(tanh-Gaussian 연속)과 발사를 직접 정합니다. 200만 스텝 학습 뒤 진화 요청 디코더 버그를 찾아 고치고, '
                '가중치를 고정한 채 새 시드로 다시 검증했습니다.',
        obs='원본 벡터 5,139차원',
        act='이동 9 · 발사 · 강화 6 · 진화 5(이산) + 회전(연속)',
        policy='장면망(인코더 → 정책/가치 MLP 128×128) + 독립 조준망 187 → 64 → 2',
        result='디코더 수정 후 새 20시드 재평가에서 기존 모델 대비 평균 <strong>1,508 → 2,902점(+92.4%)</strong>. '
               '단, 001–005의 진행 평가는 모두 버그가 있던 디코더로 측정한 값입니다.',
        headline='디코더 수정 후 +92.4%',
        extra_edges=[],
        arch=dict(cols=5, boxes=[
            ('vf', 2, 10, '가치 MLP', '128×128', 'core'),
            ('cri', 4, 10, 'Critic', '가치 1', 'head'),
            ('obs', 0, 70, '원본 벡터', '5,139', 'io'),
            ('enc', 1, 70, '개체·지형 인코더', '256', 'core'),
            ('pi', 2, 70, '정책 MLP', '128×128', 'core'),
            ('an', 3, 70, 'action_net', '이동·강화·진화', 'core'),
            ('act', 4, 70, 'Actor', '22 (결합)', 'head'),
            ('aimf', 1, 130, '조준 좌표 추출', '187', 'core'),
            ('aim', 2, 130, '독립 조준망', '64 → 2 · 회전·발사', 'core'),
        ], edges=[('obs', 'enc'), ('enc', 'pi'), ('enc', 'vf'), ('vf', 'cri'), ('pi', 'an'), ('an', 'act'),
                  ('obs', 'aimf'), ('aimf', 'aim'), ('aim', 'act', 'late')]),
    ),
    dict(
        key='gpt_training', short='조준 보조 이산', source='gpt_training/reports/lineage_prototype.html',
        name='조준 보조 이산 행동', tag='전부 이산 · 조준 보정망', tone='#c98208',
        summary='원본 5,139차원 관찰을 개체·지형 인코더(256)로 압축하고, 이동·회전 세기·발사·강화·진화를 모두 이산(MultiDiscrete 9·9·2·6·5)으로 '
                '고릅니다. 중간부터는 가까운 표적의 상대 좌표를 정렬한 조준 보정망이 회전·발사 로짓을 보정합니다. 가장 오래 이어진 계열로, '
                '모방학습 혼합(001–005) 이후 순수 PPO와 체크포인트 가중 평균까지 진행했습니다.',
        obs='원본 벡터 5,139차원',
        act='이동 9 · 회전 세기 9 · 발사 2 · 강화 6 · 진화 5, 전부 이산',
        policy='개체·지형 인코더 → 정책/가치 MLP 128×128 (+ 조준 보정망)',
        result='공개 기본 모델은 <strong>018</strong>입니다. 022의 체크포인트 가중 평균 후보는 새 20시드 평균이 1,591 → 2,058점으로 높았지만 '
               '120초 점수 p25가 865 → 650으로 떨어져 <strong>기본 모델을 교체하지 않았습니다</strong>.',
        headline='018 공개 모델 유지',
        extra_edges=[('018_reaction_calibration', '022_checkpoint_blend'), ('020_time_discount', '022_checkpoint_blend')],
        arch=dict(cols=4, boxes=[
            ('vf', 2, 10, '가치 MLP', '128×128', 'core'),
            ('cri', 3, 10, 'Critic', '가치 1', 'head'),
            ('obs', 0, 70, '원본 벡터', '5,139', 'io'),
            ('enc', 1, 70, '개체·지형 인코더', '256', 'core'),
            ('pi', 2, 70, '정책 MLP', '128×128', 'core'),
            ('act', 3, 70, 'Actor 로짓', '9·9·2·6·5 = 31', 'head'),
            ('geo', 1, 130, '조준 기하 추출', '187 + latent', 'opt'),
            ('cor', 2, 130, '조준 보정망', '64 → 11 · 0 초기화', 'opt'),
        ], edges=[('obs', 'enc'), ('enc', 'pi'), ('enc', 'vf'), ('vf', 'cri'), ('pi', 'act'),
                  ('obs', 'geo', 'opt'), ('geo', 'cor', 'opt'), ('cor', 'act', 'opt')]),
    ),
    dict(
        key='training_hurwan', short='지형 CNN', source='training_hurwan/reports/lineage_prototype.html',
        name='지형 격자 CNN', tag='지형만 CNN · 연속 → 이산', tone='#e0435c',
        summary='벡터 관찰에서 지형 격자 부분만 작은 CNN(16 → 32 채널)으로 압축하고 나머지 수치는 그대로 이어 붙이는 가장 단순한 구조입니다. '
                '연속 행동 기준선(002)에 지형 CNN을 더하고(003), 이산 행동으로 바꿔(004) 소행성 → 전투 커리큘럼을 13단계로 늘렸습니다.',
        obs='벡터 관찰(수치 + 지형 격자)',
        act='002·003 연속 → 004 이산',
        policy='SB3 기본 정책/가치 MLP 64×64 (+ 003·004 지형 CNN)',
        result='네 run은 체크포인트를 공유하지 않는 독립 실험입니다. <strong>004</strong>가 13단계 커리큘럼으로 가장 깊게 진행됐지만 '
               '중단(interrupted) 상태로 끝났고, 005(전투 + 소행성)는 설계만 있고 학습을 돌리지 않았습니다.',
        headline='004 · 13단계 커리큘럼',
        extra_edges=[],
        arch=dict(cols=5, boxes=[
            ('obs', 0, 70, '벡터 관찰', '수치 + 지형 격자', 'io'),
            ('num', 1, 10, '수치 특징', '그대로 통과', 'core'),
            ('cnn', 1, 130, '지형 격자 CNN', '16→32 · 003·004', 'opt'),
            ('cat', 2, 70, '특징 결합', 'concat', 'core'),
            ('pi', 3, 10, '정책 MLP', '64×64 · 기본', 'core'),
            ('vf', 3, 130, '가치 MLP', '64×64 · 기본', 'core'),
            ('act', 4, 10, 'Actor', '연속 → 이산(004)', 'head'),
            ('cri', 4, 130, 'Critic', '가치 1', 'head'),
        ], edges=[('obs', 'num'), ('obs', 'cnn', 'opt'), ('num', 'cat'), ('cnn', 'cat', 'opt'), ('cat', 'pi'), ('cat', 'vf'),
                  ('pi', 'act'), ('vf', 'cri')]),
    ),
]

ICON_MOON = '<svg class="ico-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
ICON_SUN = '<svg class="ico-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
ICON_MENU = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>'
ICON_CLOSE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>'
THEME_BOOT = "<script>try{var t=localStorage.getItem('szr-theme');if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}</script>"


def esc(value):
    return html.escape(str(value), quote=True)


def text_of(element):
    return element.get_text(' ', strip=True) if element else ''


def inner(element):
    return ''.join(str(child) for child in element.contents).strip()


def plain(fragment):
    return html.unescape(re.sub(r'<[^>]+>', '', fragment)).strip()


def text_width(value, size, mono=False):
    width = 0.0
    for char in value:
        if mono:
            width += 0.62
        elif 'ᄀ' <= char or char in '→←—…':
            width += 0.98
        elif char.isdigit():
            width += 0.6
        else:
            width += 0.56
    return width * size


def clip(value, size, limit, mono=False):
    if text_width(value, size, mono) <= limit:
        return value
    while value and text_width(value + '…', size, mono) > limit:
        value = value[:-1]
    return value.rstrip() + '…'


def split_title(title):
    match = re.match(r'^(\d{3})[_\s·-]+(.+)$', title)
    return (match.group(1), match.group(2)) if match else ('', title)


def percent(style, name):
    match = re.search(rf'(?<![-\w]){name}\s*:\s*([\d.]+)%', style or '')
    return float(match.group(1)) if match else None


def status_class(classes):
    return next((name for name in classes if name in STATUS_CLASSES), 'base')


class Context:
    """한 계열 페이지를 변환하는 동안 이미지 복사와 경고를 모은다."""

    def __init__(self, key, source_dir):
        self.key = key
        self.source_dir = source_dir
        self.images = {}
        self.warnings = []

    def image(self, src):
        path = (self.source_dir / src).resolve()
        if not path.exists():
            self.warnings.append(f'missing image: {src}')
            return src
        name = '__'.join(part for part in Path(src).parts if part not in ('..', '.'))
        self.images[path] = OUT / 'assets' / 'img' / self.key / name
        return f'../assets/img/{self.key}/{name}'

    def normalize(self, element):
        for tag in [element, *element.find_all(True)]:
            style = tag.get('style')
            if style is not None:
                color = re.search(r'color\s*:\s*var\(--(good|bad|warn|accent)\)', style)
                if color:
                    tag['class'] = list(tag.get('class', [])) + ['c-' + color.group(1)]
                del tag['style']
            if tag.name == 'a':
                href = tag.get('href', '')
                if href.endswith('reports/index.html'):
                    tag['href'] = '../index.html'
                elif not href.startswith('#'):
                    self.warnings.append(f'unlinked external href: {href}')
                    tag.name = 'span'
                    tag.attrs = {'class': ['ref']}
            elif tag.name == 'img' and tag.get('src'):
                tag['src'] = self.image(tag['src'])
        return element


# ---------------------------------------------------------------- parsing

def parse_table(table):
    heads = [inner(th) for th in table.select('thead th')]
    rows = table.select('tbody tr') or [tr for tr in table.find_all('tr') if tr.find('td')]
    if not heads:
        first = table.find('tr')
        if first and first.find('th') and not first.find('td'):
            heads = [inner(th) for th in first.find_all('th')]
    return dict(heads=heads, rows=[dict(cls=[c for c in tr.get('class', []) if c in ('peak', 'crash')],
                                        cells=[inner(td) for td in tr.find_all(['td', 'th'])]) for tr in rows])


def parse_chart_row(row):
    box = row.select_one('.pbar-box')
    marks = {}
    for tag in row.select('.pbar-tick'):
        kind = next(c for c in tag['class'] if c in ('p25', 'p75', 'med', 'mean', 'max'))
        marks.setdefault(kind, {})['pos'] = percent(tag.get('style'), 'left')
    for tag in row.select('.pbar-tag'):
        kind = next(c for c in tag['class'] if c in ('p25', 'p75', 'med', 'mean', 'max'))
        mark = marks.setdefault(kind, {})
        if mark.get('pos') is None:
            mark['pos'] = percent(tag.get('style'), 'left')
        mark['text'] = text_of(tag)
    return dict(label=text_of(row.select_one('.pbar-label')),
                left=percent(box.get('style'), 'left') if box else None,
                width=percent(box.get('style'), 'width') if box else None, marks=marks)


def parse_charts(section, ctx):
    charts, current = [], None
    for child in section.find_all(recursive=False):
        classes = child.get('class', [])
        if child.name in ('h3', 'h4', 'h5'):
            current = dict(type='chart', title=text_of(child), rows=[], notes=[])
            charts.append(current)
        elif 'pbar-row' in classes:
            if current is None:
                current = dict(type='chart', title='분포', rows=[], notes=[])
                charts.append(current)
            current['rows'].append(parse_chart_row(child))
        elif 'pbar-legend' not in classes:
            note = str(ctx.normalize(child))
            if current is None:
                charts.append(dict(type='text', html=note))
            else:
                current['notes'].append(note)
    return charts


def parse_tables(wrapper, caption, ctx):
    blocks = []
    for child in wrapper.find_all(recursive=False):
        if child.name in ('h3', 'h4', 'h5'):
            caption = inner(ctx.normalize(child))
        elif child.name == 'table':
            ctx.normalize(child)
            kind = 'seed' if 'seedtable' in child.get('class', []) else 'table'
            blocks.append(dict(type=kind, table=parse_table(child), caption=caption))
            caption = None
        else:
            blocks.append(dict(type='text', html=str(ctx.normalize(child))))
    return blocks


def parse_variable(element, ctx):
    ctx.normalize(element)
    first = next((c for c in element.contents if not (isinstance(c, str) and not c.strip())), None)
    if isinstance(first, Tag) and first.name in ('b', 'strong') and '변경' in first.get_text():
        first.decompose()
    return re.sub(r'^\s*[:：]\s*', '', inner(element))


def parse_cfgdiff(element, ctx):
    ctx.normalize(element)
    table = element.find('table')
    extras = [str(c) for c in element.find_all(recursive=False) if c is not table]
    if table is None:
        return [dict(type='text', html=inner(element))]
    rows = table.select('tbody tr')
    cells = [[inner(td) for td in tr.find_all('td')] for tr in rows]
    if not cells or any(len(row) != 3 for row in cells):
        return [dict(type='table', table=parse_table(table), caption='설정 변경')] + [dict(type='text', html=e) for e in extras]
    heads = [text_of(th) for th in table.select('thead th')] or ['설정', '부모', '이번 실험']
    return [dict(type='cfgdiff', heads=heads, rows=cells, extras=extras)]


def parse_blocks(container, ctx):
    blocks, caption = [], None
    for child in container.find_all(recursive=False):
        classes = child.get('class', [])
        first = classes[0] if classes else ''
        if first == 'two-col':
            blocks += parse_blocks(child, ctx)
        elif child.name in ('h3', 'h4', 'h5') and child.find_next_sibling() is not None \
                and (child.find_next_sibling().get('class') or [''])[0] in ('table-wrap', 'table-scroll'):
            caption = inner(ctx.normalize(child))
        elif first == 'variable':
            blocks.append(dict(type='variable', html=parse_variable(child, ctx)))
        elif first == 'cfgdiff':
            blocks += parse_cfgdiff(child, ctx)
        elif first in ('table-wrap', 'table-scroll'):
            blocks += parse_tables(child, caption, ctx)
            caption = None
        elif first == 'shots':
            figures = []
            for figure in child.select('figure'):
                ctx.normalize(figure)
                image = figure.find('img')
                caption_tag = figure.find('figcaption')
                figures.append(dict(src=image['src'], alt=image.get('alt', ''), caption=inner(caption_tag) if caption_tag else ''))
            blocks.append(dict(type='shots', figures=figures))
        elif first == 'pbar-section':
            blocks += parse_charts(child, ctx)
        elif first == 'confirm':
            blocks.append(dict(type='confirm', html=inner(ctx.normalize(child))))
        elif first == 'note':
            blocks.append(dict(type='alert' if 'alert' in classes else 'note', html=inner(ctx.normalize(child))))
        elif child.name == 'span' and 'badge' in classes:
            blocks.append(dict(type='badge', text=text_of(child)))
        elif child.name in ('h1', 'h2'):
            blocks.append(dict(type=child.name, html=inner(ctx.normalize(child))))
        else:
            blocks.append(dict(type='text', html=str(ctx.normalize(child))))
    return blocks


def parse_page(section, ctx):
    page = dict(id=section['id'], context=None, card=None, hero=None, root_rows=[], blocks=[])
    for child in section.find_all(recursive=False):
        classes = child.get('class', [])
        if 'siblings' in classes:
            page['context'] = inner(ctx.normalize(child))
        elif 'pagenav' in classes:
            continue
        elif 'card' in classes:
            head = child.select_one('.card-head')
            status = head.select_one('.status') if head else None
            page['card'] = dict(
                title=text_of(head.find('h3')) if head else page['id'],
                steps=text_of(head.select_one('.id')) if head else '',
                status=text_of(status), cls=status_class(status.get('class', [])) if status else 'base')
            page['blocks'] += parse_blocks(child.select_one('.card-body') or child, ctx)
        elif 'hero' in classes:
            page['hero'] = parse_blocks(child, ctx)
        elif 'root-row' in classes:
            page['root_rows'].append(inner(ctx.normalize(child)))
        else:
            page['blocks'] += parse_blocks(BeautifulSoup(f'<div>{child}</div>', 'html.parser').div, ctx)
    return page


def path_points(d):
    subpaths, current, x, y = [], None, 0.0, 0.0
    for command, numbers in re.findall(r'([MHVL])\s*([-\d.,\s]+)', d):
        values = [float(n) for n in re.split(r'[\s,]+', numbers.strip()) if n]
        if command == 'M':
            x, y = values[0], values[1]
            current = [(x, y)]
            subpaths.append(current)
        elif command == 'H':
            x = values[0]
            current.append((x, y))
        elif command == 'V':
            y = values[0]
            current.append((x, y))
        elif command == 'L':
            x, y = values[0], values[1]
            current.append((x, y))
    return subpaths


def parse_tree(soup):
    """이전 분기도 SVG에서 노드 문구와 부모→자식 간선을 복원한다."""
    svg = soup.select_one('.treenav svg')
    nodes, boxes = {}, []
    for anchor in svg.select('a.tree-node'):
        node_id = anchor.get('data-node') or anchor.get('href', '')[1:]
        rect = anchor.select_one('rect.svg-node')
        tags = [text_of(t) for t in anchor.select('text.svg-tag') if 'cur-badge' not in t.get('class', [])]
        nodes[node_id] = dict(title=text_of(anchor.select_one('text.svg-text')), sub=text_of(anchor.select_one('text.svg-sub')),
                              tag=' · '.join(tags), cls=status_class(rect.get('class', [])))
        boxes.append((node_id, float(rect['x']), float(rect['y']), float(rect['width']), float(rect['height'])))
    segments = [(a, b) for path in svg.select('path.svg-edge') for sub in path_points(path['d']) for a, b in zip(sub, sub[1:])]
    parent = list(range(len(segments)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def touches(point, segment):
        (x1, y1), (x2, y2) = segment
        return min(x1, x2) - .5 <= point[0] <= max(x1, x2) + .5 and min(y1, y2) - .5 <= point[1] <= max(y1, y2) + .5

    for i, s in enumerate(segments):
        for j in range(i + 1, len(segments)):
            t = segments[j]
            if touches(s[0], t) or touches(s[1], t) or touches(t[0], s) or touches(t[1], s):
                parent[find(i)] = find(j)
    groups = {}
    for i, s in enumerate(segments):
        groups.setdefault(find(i), []).extend(s)
    edges = []
    for points in groups.values():
        sources = [b[0] for b in boxes if any(abs(p[0] - (b[1] + b[3])) < 3 and b[2] - 2 <= p[1] <= b[2] + b[4] + 2 for p in points)]
        targets = [b[0] for b in boxes if any(abs(p[0] - b[1]) < 3 and b[2] - 2 <= p[1] <= b[2] + b[4] + 2 for p in points)]
        edges += [(a, b) for a in sources for b in targets if a != b and (a, b) not in edges]
    return nodes, edges


def parse_approach(spec):
    source = RL / spec['source']
    ctx = Context(spec['key'], source.parent)
    soup = BeautifulSoup(source.read_text(encoding='utf-8'), 'html.parser')
    tree_nodes, edges = parse_tree(soup)
    pages = [parse_page(section, ctx) for section in soup.select('section.page')]
    edges += [e for e in spec['extra_edges'] if e not in edges]
    runs = {}
    for page in pages:
        match = re.match(r'^s(\d{3})__', page['id'])
        if match:
            runs.setdefault('ov_s' + match.group(1), []).append(page['id'])
    for root, sessions in runs.items():
        chain = [root] + sessions
        edges += [(a, b) for a, b in zip(chain, chain[1:]) if (a, b) not in edges]
    by_id = {p['id']: p for p in pages}
    flow_ids = [p['id'] for p in pages if p['id'] in tree_nodes or any(p['id'] in e for e in edges)]
    info = {}
    for node_id in flow_ids:
        page = by_id[node_id]
        tree = tree_nodes.get(node_id)
        if tree:
            info[node_id] = dict(tree)
        else:
            card = page['card'] or {}
            variable = next((plain(b['html']) for b in page['blocks'] if b['type'] == 'variable'), '')
            info[node_id] = dict(title=card.get('title', node_id), sub=variable, tag=card.get('status', ''), cls=card.get('cls', 'base'))
    return dict(spec=spec, ctx=ctx, pages=pages, by_id=by_id, flow_ids=flow_ids, edges=edges, info=info)


# ---------------------------------------------------------------- layout

NODE_W, NODE_H, GAP_X, GAP_Y = 256, 78, 48, 18


def layout_flow(node_ids, edges):
    parents = {n: [] for n in node_ids}
    children = {n: [] for n in node_ids}
    for a, b in edges:
        children[a].append(b)
        parents[b].append(a)
    order, ready, seen = [], [n for n in node_ids if not parents[n]], set()
    remaining = {n: len(parents[n]) for n in node_ids}
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in children[node]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    depth = {}
    for node in order:
        depth[node] = max((depth[p] + 1 for p in parents[node]), default=0)
    sizes = {}

    def size(node):
        if node not in sizes:
            sizes[node] = 1 + sum(size(c) for c in children[node])
        return sizes[node]

    row, used = {}, set()

    def place(node, wanted):
        slot = wanted
        while (depth[node], slot) in used:
            slot += 1
        row[node] = slot
        used.add((depth[node], slot))
        ranked = sorted(children[node], key=lambda c: (-size(c), order.index(c)))
        for index, child in enumerate(ranked):
            if child not in row and all(p in row for p in parents[child]):
                place(child, slot + index)

    for node in order:
        if not parents[node] and node not in row:
            place(node, max((r for _, r in used), default=-1) + 1)
    return order, parents, depth, row


def flow_svg(model):
    node_ids, edges, info = model['flow_ids'], model['edges'], model['info']
    order, parents, depth, row = layout_flow(node_ids, edges)
    pitch_x, pitch_y = NODE_W + GAP_X, NODE_H + GAP_Y
    position = {n: (2 + depth[n] * pitch_x, 2 + row[n] * pitch_y) for n in node_ids}
    max_row = max(row.values())
    width = 4 + max(depth.values()) * pitch_x + NODE_W
    lane = 2 + (max_row + 1) * pitch_y + 4
    parts = ['<defs><marker id="fa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" class="edge-head"/></marker>'
             '<marker id="fa-m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" class="edge-head merge"/></marker></defs>']
    lanes_used = 0
    for a, b in edges:
        (ax, ay), (bx, by) = position[a], position[b]
        sx, sy, ex, ey = ax + NODE_W, ay + NODE_H / 2, bx - 1, by + NODE_H / 2
        merge = len(parents[b]) > 1
        if depth[b] - depth[a] > 1:
            y = lane + lanes_used * 12
            lanes_used += 1
            d = (f'M{sx},{sy} H{sx + 10} Q{sx + 20},{sy} {sx + 20},{sy + 10} V{y - 10} Q{sx + 20},{y} {sx + 30},{y} '
                 f'H{ex - 30} Q{ex - 20},{y} {ex - 20},{y - 10} V{ey + 10} Q{ex - 20},{ey} {ex - 10},{ey} H{ex}')
        elif sy == ey:
            d = f'M{sx},{sy} H{ex}'
        else:
            d = f'M{sx},{sy} C{sx + GAP_X * .55},{sy} {ex - GAP_X * .55},{ey} {ex},{ey}'
        parts.append(f'<path class="edge{" merge" if merge else ""}" d="{d}" marker-end="url(#{"fa-m" if merge else "fa"})"/>')
    height = (lane + lanes_used * 12 + 6) if lanes_used else (2 + (max_row + 1) * pitch_y - GAP_Y + 2)
    for node in order:
        x, y = position[node]
        item = info[node]
        number, name = split_title(item['title'])
        tip = ' — '.join(v for v in (item['title'], item['sub'], item['tag']) if v)
        title_x = x + 18
        texts = []
        if number:
            texts.append(f'<text class="fn-num" x="{title_x}" y="{y + 25}">{esc(number)}</text>')
            title_x += text_width(number, 12.5, mono=True) + 7
        texts.append(f'<text class="fn-title" x="{title_x}" y="{y + 25}">{esc(clip(name, 13, x + NODE_W - 44 - title_x))}</text>')
        if item['sub']:
            texts.append(f'<text class="fn-sub" x="{x + 18}" y="{y + 45}">{esc(clip(item["sub"], 11.5, NODE_W - 30))}</text>')
        if item['tag']:
            texts.append(f'<text class="fn-tag" x="{x + 18}" y="{y + 64}">{esc(clip(item["tag"], 11, NODE_W - 30))}</text>')
        parts.append(
            f'<a href="#{esc(node)}" class="fn s-{item["cls"]}" data-target="{esc(node)}" data-x="{x}"><title>{esc(tip)}</title>'
            f'<rect class="fn-box" x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="12"/>'
            f'<rect class="fn-bar" x="{x + 7}" y="{y + 12}" width="3.5" height="{NODE_H - 24}" rx="1.75"/>'
            + ''.join(texts) +
            f'<text class="fn-here" x="{x + NODE_W - 12}" y="{y + 24}" text-anchor="end">현재</text></a>')
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="실험 계보 그래프">{"".join(parts)}</svg>', order, row


def arch_svg(spec, key):
    box_w, box_h, pitch = 132, 46, 156
    position = {b[0]: (10 + b[1] * pitch, b[2]) for b in spec['boxes']}
    kinds = {b[0]: b[5] for b in spec['boxes']}
    width = 20 + (spec['cols'] - 1) * pitch + box_w
    height = max(b[2] for b in spec['boxes']) + box_h + 10
    parts = [f'<defs><marker id="ah-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>'
             f'<marker id="aho-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" class="ah opt"/></marker></defs>']
    for edge in spec['edges']:
        a, b, style = edge[0], edge[1], (edge[2] if len(edge) > 2 else '')
        (ax, ay), (bx, by) = position[a], position[b]
        sx, sy, ex, ey = ax + box_w, ay + box_h / 2, bx - 1, by + box_h / 2
        optional = style == 'opt' or kinds[a] == 'opt' or kinds[b] == 'opt'
        if style == 'late':
            d = f'M{sx},{sy} H{ex - 26} C{ex - 12},{sy} {ex - 12},{ey} {ex},{ey}'
        elif sy == ey:
            d = f'M{sx},{sy} H{ex}'
        else:
            mid = (sx + ex) / 2
            d = f'M{sx},{sy} C{mid},{sy} {mid},{ey} {ex},{ey}'
        parts.append(f'<path class="ae{" opt" if optional else ""}" d="{d}" marker-end="url(#{"aho" if optional else "ah"}-{key})"/>')
    for box_id, _, _, label, sub, kind in spec['boxes']:
        x, y = position[box_id]
        parts.append(f'<rect class="ab {kind}" x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="10"/>'
                     f'<text class="at" x="{x + box_w / 2}" y="{y + 19}" text-anchor="middle">{esc(label)}</text>'
                     f'<text class="as" x="{x + box_w / 2}" y="{y + 35}" text-anchor="middle">{esc(sub)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="신경망 구조">{"".join(parts)}</svg>'


# ---------------------------------------------------------------- rendering

def number_value(value):
    match = re.search(r'[-−]?[\d,]*\.?\d+', value or '')
    return float(match.group(0).replace(',', '').replace('−', '-')) if match else None


def chart_scale(chart):
    match = re.search(r'척도\s*0\s*~\s*([\d,]+)', chart['title'])
    if match:
        return float(match.group(1).replace(',', ''))
    estimates = []
    for row in chart['rows']:
        for kind, mark in row['marks'].items():
            value = number_value((mark.get('text') or '').replace(kind, '', 1))
            if value and mark.get('pos') and mark['pos'] > 3:
                estimates.append(value / (mark['pos'] / 100))
    return round(statistics.median(estimates)) if estimates else None


def axis_label(value):
    return f'{value:,.0f}' if value >= 100 or float(value).is_integer() else f'{value:.1f}'


def axis_ticks(scale):
    raw = scale / 4
    magnitude = 10 ** int(f'{raw:e}'.split('e')[1])
    step = next(m * magnitude for m in (1, 2, 2.5, 3, 5, 10) if m * magnitude >= raw)
    count = int(scale / step + 1e-9)
    return [k * step for k in range(count + 1)]


def place_labels(items, low, high):
    placed = []
    for x, label, kind in sorted(items, key=lambda item: item[0]):
        width = text_width(label, 10.5, mono=True) + 6
        center = x
        if placed and center - width / 2 < placed[-1][1] + placed[-1][2] / 2 + 4:
            center = placed[-1][1] + placed[-1][2] / 2 + 4 + width / 2
        placed.append([label, center, width, kind])
    for index in range(len(placed) - 1, -1, -1):
        limit = high if index == len(placed) - 1 else placed[index + 1][1] - placed[index + 1][2] / 2 - 4
        if placed[index][1] + placed[index][2] / 2 > limit:
            placed[index][1] = limit - placed[index][2] / 2
    for item in placed:
        if item[1] - item[2] / 2 < low:
            item[1] = low + item[2] / 2
    return placed


def chart_svg(chart):
    longest = max((text_width(r['label'], 12, mono=True) for r in chart['rows']), default=60)
    label_x = 16 + min(170, max(88, longest))
    width, x0, row_h, top = 760, label_x + 12, 62, 36
    x1 = width - 28
    plot = x1 - x0
    scale = chart_scale(chart)
    unit = 's' if '생존' in chart['title'] else ''
    rows = chart['rows']
    height = top + len(rows) * row_h + 4

    def at(pos):
        return x0 + max(0.0, min(100.0, pos or 0.0)) / 100 * plot

    parts = []
    for value in (axis_ticks(scale) if scale else []):
        x = x0 + value / scale * plot
        parts.append(f'<line class="cx-grid" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{height - 4}"/>')
        parts.append(f'<text class="cx-axis" x="{x:.1f}" y="{top - 16}" text-anchor="middle">{axis_label(value)}{unit}</text>')
    for index, row in enumerate(rows):
        cy = top + index * row_h + 30
        marks = row['marks']
        tip = row['label'] + ': ' + ' · '.join(m['text'] for k in ('p25', 'med', 'mean', 'p75', 'max') if (m := marks.get(k)) and m.get('text'))
        g = [f'<g><title>{esc(tip)}</title>',
             f'<text class="cx-row" x="{label_x}" y="{cy + 4}" text-anchor="end">{esc(clip(row["label"], 12, label_x - 16, mono=True))}</text>']
        end = marks.get('max', {}).get('pos')
        g.append(f'<line class="cx-track" x1="{x0}" y1="{cy}" x2="{at(end if end is not None else 100):.1f}" y2="{cy}"/>')
        if row['left'] is not None and row['width'] is not None:
            g.append(f'<rect class="cx-box" x="{at(row["left"]):.1f}" y="{cy - 8}" width="{max(2.5, row["width"] / 100 * plot):.1f}" height="16" rx="4"/>')
        if 'max' in marks and marks['max'].get('pos') is not None:
            x = at(marks['max']['pos'])
            g.append(f'<line class="cx-max" x1="{x:.1f}" y1="{cy - 11}" x2="{x:.1f}" y2="{cy + 11}"/>')
        if 'med' in marks and marks['med'].get('pos') is not None:
            x = at(marks['med']['pos'])
            g.append(f'<line class="cx-med" x1="{x:.1f}" y1="{cy - 11}" x2="{x:.1f}" y2="{cy + 11}"/>')
        if 'mean' in marks and marks['mean'].get('pos') is not None:
            x = at(marks['mean']['pos'])
            g.append(f'<path class="cx-mean" d="M{x:.1f},{cy - 7} L{x + 7:.1f},{cy} L{x:.1f},{cy + 7} L{x - 7:.1f},{cy} Z"/>')
        for kinds, y in ((('p25', 'p75', 'max'), cy - 16), (('med', 'mean'), cy + 26)):
            items = [(at(marks[k]['pos']), marks[k]['text'], k) for k in kinds
                     if k in marks and marks[k].get('pos') is not None and marks[k].get('text')]
            for label, center, _, kind in place_labels(items, 40, width - 2):
                g.append(f'<text class="cx-lb {kind}" x="{center:.1f}" y="{y}" text-anchor="middle">{esc(label)}</text>')
        g.append('</g>')
        parts.append(''.join(g))
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(chart["title"])}">{"".join(parts)}</svg>'


CHART_LEGEND = (
    '<div class="chart-legend">'
    '<span><svg width="20" height="12"><rect x="1" y="2" width="18" height="8" rx="2" class="cx-box"/></svg>p25–p75 구간</span>'
    '<span><svg width="10" height="16"><line x1="5" y1="2" x2="5" y2="14" class="cx-med"/></svg>median</span>'
    '<span><svg width="14" height="14"><path d="M7,1 L13,7 L7,13 L1,7 Z" class="cx-mean"/></svg>mean</span>'
    '<span><svg width="10" height="16"><line x1="5" y1="2" x2="5" y2="14" class="cx-max"/></svg>max(단일 시드 최고)</span>'
    '</div>')


def is_number(fragment):
    value = plain(fragment)
    return bool(re.fullmatch(r'[-+−]?[\d,]*\.?\d+\s*[a-zA-Z%초점]*(\s*[(\[][^)\]]*[)\]])?', value))


def render_table(table, caption=None, seed=False):
    heads, rows = table['heads'], table['rows']
    wins = set()
    if seed and len(rows) >= 2:
        columns = max(len(r['cells']) for r in rows)
        for column in range(1, columns):
            values = [(i, number_value(plain(r['cells'][column]))) for i, r in enumerate(rows) if column < len(r['cells'])]
            numbers = [v for _, v in values if v is not None]
            if len(numbers) >= 2 and len(set(numbers)) > 1:
                top = max(numbers)
                wins |= {(i, column) for i, v in values if v == top}
    numeric_columns = set()
    for column in range(1, len(heads)):
        cells = [r['cells'][column] for r in rows if column < len(r['cells'])]
        if cells and sum(is_number(c) for c in cells) >= max(1, len(cells) * .6):
            numeric_columns.add(column)
    out = ['<div class="tbl-wrap">']
    if caption:
        out.append(f'<div class="tbl-cap">{caption}</div>')
    out.append('<table class="tbl">')
    if heads:
        out.append('<thead><tr>' + ''.join(f'<th class="{"num" if i in numeric_columns else ""}">{h}</th>' for i, h in enumerate(heads)) + '</tr></thead>')
    out.append('<tbody>')
    for i, row in enumerate(rows):
        cells = ''.join(
            f'<td class="{" ".join(c for c in ("num" if (j in numeric_columns or (seed and j > 0)) else "", "win" if (i, j) in wins else "") if c)}">{cell}</td>'
            for j, cell in enumerate(row['cells']))
        out.append(f'<tr class="{" ".join(row["cls"])}">{cells}</tr>')
    out.append('</tbody></table></div>')
    return ''.join(out)


def render_kpis(blocks):
    for block in blocks:
        if block['type'] != 'table':
            continue
        heads = [plain(h).lower() for h in block['table']['heads']]
        peak = next((r for r in block['table']['rows'] if 'peak' in r['cls']), None)
        if not peak or not heads or heads[0] != 'label':
            continue
        tiles = ''.join(f'<div class="panel kpi"><div class="kpi-label">{block["table"]["heads"][j]}</div><div class="kpi-val">{peak["cells"][j]}</div></div>'
                        for j in range(1, min(len(heads), len(peak['cells']))))
        return f'<div class="kpi-cap">표에서 <b>BEST</b>로 표시된 체크포인트 · {peak["cells"][0]}</div><div class="kpis">{tiles}</div>'
    return ''


def render_chart(chart):
    pieces = [part.strip() for part in chart['title'].split('·')]
    meta = ''.join(f'<span class="chip">{esc(p)}</span>' for p in pieces[1:] if p)
    notes = ''.join(f'<div class="chart-note">{n}</div>' for n in chart['notes'])
    body = chart_svg(chart) if chart['rows'] else ''
    return (f'<div class="panel chart"><div class="chart-head"><h3>{esc(pieces[0])}</h3><div class="chart-meta">{meta}</div></div>'
            f'{body}{notes}</div>')


def render_block(block):
    kind = block['type']
    if kind == 'variable':
        return f'<div class="change"><span class="change-ico">Δ</span><div>{block["html"]}</div></div>'
    if kind == 'cfgdiff':
        heads = block['heads']
        rows = ''.join(f'<div class="diff-row"><span class="diff-key">{k}</span><span class="diff-old">{a}</span>'
                       f'<span class="diff-arrow">→</span><span class="diff-new">{b}</span></div>' for k, a, b in block['rows'])
        foot = ''.join(f'<div class="diff-foot">{e}</div>' for e in block['extras'])
        head = (f'<div class="diff-head"><span>{esc(heads[0])}</span><span>{esc(heads[1] if len(heads) > 1 else "부모")}</span><span></span>'
                f'<span>{esc(heads[2] if len(heads) > 2 else "이번 실험")}</span></div>')
        return f'<div class="panel diff">{head}{rows}{foot}</div>'
    if kind in ('table', 'seed'):
        return render_table(block['table'], block.get('caption'), seed=kind == 'seed')
    if kind == 'chart':
        return render_chart(block)
    if kind == 'shots':
        figures = ''.join(f'<figure class="panel shot"><button class="shot-btn" type="button" data-full="{esc(f["src"])}" aria-label="크게 보기">'
                          f'<img src="{esc(f["src"])}" alt="{esc(f["alt"])}" loading="lazy"></button><figcaption>{f["caption"]}</figcaption></figure>'
                          for f in block['figures'])
        return f'<div class="shots">{figures}</div>'
    if kind == 'confirm':
        return f'<div class="panel heldout">{block["html"]}</div>'
    if kind in ('note', 'alert'):
        warn = kind == 'alert'
        return (f'<div class="callout{" warn" if warn else ""}"><span class="callout-ico">{"!" if warn else "i"}</span>'
                f'<div class="callout-body">{block["html"]}</div></div>')
    if kind == 'badge':
        return f'<span class="chip">{esc(block["text"])}</span>'
    if kind in ('h1', 'h2'):
        return f'<h3 class="question">{block["html"]}</h3>'
    return f'<div class="prose">{block["html"]}</div>'


GROUPS = [('alerts', None), ('change', '무엇을 바꿨나'), ('table', '평가 결과'), ('chart', '점수·생존 분포'),
          ('shots', '실제 플레이 화면'), ('heldout', 'held-out 검증'), ('notes', '해석과 메모')]
GROUP_OF = dict(alert='alerts', variable='change', cfgdiff='change', table='table', chart='chart', shots='shots',
                confirm='heldout', seed='heldout', note='notes')


def render_sections(blocks):
    grouped, last = {g: [] for g, _ in GROUPS}, 'change'
    for block in blocks:
        group = GROUP_OF.get(block['type'], last)
        grouped[group].append(block)
        if block['type'] in GROUP_OF:
            last = group
    out, number = [], 0
    for group, title in GROUPS:
        items = grouped[group]
        if not items:
            continue
        body = ''.join(render_block(b) for b in items)
        if group == 'table':
            body = render_kpis(items) + body
        if group == 'chart':
            body = CHART_LEGEND + body
        if title is None:
            out.append(f'<div class="stack" style="margin-bottom:8px">{body}</div>')
            continue
        number += 1
        out.append(f'<div class="sec"><h2 class="sec-title"><span class="sec-num">{number:02d}</span>{title}</h2><div class="stack">{body}</div></div>')
    return ''.join(out)


def page_title(model, page_id):
    page = model['by_id'][page_id]
    if page['card']:
        return page['card']['title']
    if page_id == 'overview':
        return '개요'
    heading = next((b for b in page['hero'] or [] if b['type'] in ('h1', 'h2')), None)
    if heading:
        return plain(heading['html'])
    return model['info'].get(page_id, {}).get('title', page_id)


def build_navigation(model, flow_order, rows):
    edges = model['edges']
    components, assigned = [], {}
    for node in flow_order:
        if node in assigned:
            continue
        stack, members = [node], []
        while stack:
            current = stack.pop()
            if current in assigned:
                continue
            assigned[current] = len(components)
            members.append(current)
            stack += [b for a, b in edges if a == current] + [a for a, b in edges if b == current]
        components.append(sorted(members, key=flow_order.index))
    extras = [p['id'] for p in model['pages'] if p['id'] != 'overview' and p['id'] not in assigned]
    groups = []
    if extras:
        groups.append(('이전 단계', extras))
    for members in components:
        title = '실험 계보' if len(components) == 1 else page_title(model, members[0])
        groups.append((title, members))
    sequence = ['overview'] + [m for _, members in groups for m in members]
    return groups, sequence


def page_status(model, page_id):
    page = model['by_id'][page_id]
    if page_id in model['info']:
        return model['info'][page_id]['cls']
    return page['card']['cls'] if page['card'] else 'base'


def render_sidebar(model, groups):
    spec = model['spec']
    out = [f'<aside class="sidebar" aria-label="실험 목록"><div class="side-head"><span class="eyebrow">{esc(spec["tag"])}</span><b>{esc(spec["name"])}</b></div>',
           '<a class="side-link" href="#overview" data-target="overview"><span class="dot s-base"></span>개요</a>']
    for title, members in groups:
        out.append(f'<div class="side-group"><div class="side-title">{esc(title)}</div>')
        for member in members:
            number, name = split_title(page_title(model, member))
            if member.startswith('ov_'):
                number, name = '', '개요 · ' + name
            out.append(f'<a class="side-item" href="#{esc(member)}" data-target="{esc(member)}"><span class="dot s-{page_status(model, member)}"></span>'
                       f'<span class="side-num">{esc(number)}</span><span class="side-name" title="{esc(name)}">{esc(name)}</span></a>')
        out.append('</div>')
    out.append('</aside>')
    return ''.join(out)


def render_pn(model, sequence, page_id):
    index = sequence.index(page_id)
    cells = []
    for offset, label, css in ((-1, '이전', 'prev'), (1, '다음', 'next')):
        target = index + offset
        if 0 <= target < len(sequence):
            cells.append(f'<a class="panel {css}" href="#{esc(sequence[target])}"><span class="pn-lbl">{"← " if offset < 0 else ""}{label}{" →" if offset > 0 else ""}</span>'
                         f'<span class="pn-t">{esc(page_title(model, sequence[target]))}</span></a>')
        else:
            cells.append('<span class="blank"></span>')
    return f'<nav class="pn">{"".join(cells)}</nav>'


def exp_cards(model, members):
    cards = []
    for member in members:
        page = model['by_id'][member]
        number, name = split_title(page_title(model, member))
        info = model['info'].get(member, {})
        status = page['card']['status'] if page['card'] else info.get('tag', '')
        desc = next((plain(b['html']) for b in page['blocks'] if b['type'] == 'variable'), '') or info.get('sub', '')
        cards.append(f'<a class="panel exp-card" href="#{esc(member)}"><div class="top"><span class="num">{esc(number or "run")}</span>'
                     f'<span class="pill s-{page_status(model, member)}">{esc(clip(status, 12, 150))}</span></div>'
                     f'<div class="name">{esc(name)}</div><div class="desc">{esc(desc)}</div></a>')
    return f'<div class="exp-grid">{"".join(cards)}</div>'


def render_overview(model, groups, sequence):
    spec = model['spec']
    page = model['by_id']['overview']
    hero = page['hero'] or []
    badges = ''.join(f'<span class="chip">{esc(b["text"])}</span>' for b in hero if b['type'] == 'badge')
    question = next((b['html'] for b in hero if b['type'] in ('h1', 'h2')), '')
    prose = ''.join(render_block(b) for b in hero if b['type'] not in ('badge', 'h1', 'h2'))
    roots = ''.join(f'<div class="callout"><span class="callout-ico">↺</span><div class="callout-body">{r}</div></div>' for r in page['root_rows'])
    count = len(sequence) - 1
    listing = ''.join(f'<div class="group-title">{esc(title)}</div>{exp_cards(model, members)}' for title, members in groups)
    return (
        f'<section class="page" data-page="overview" data-title="개요">'
        f'<div class="ov-hero"><span class="eyebrow">{esc(spec["tag"])}</span><h1>{esc(spec["name"])}</h1>'
        f'<p class="lead">{esc(spec["summary"])}</p>'
        f'<div class="facts"><span class="chip"><b>{count}</b>개 페이지</span><span class="chip">관찰 · {esc(spec["obs"])}</span>{badges}</div></div>'
        f'<div class="grid-2"><div class="panel panel-pad arch"><div class="panel-title">신경망 구조</div>{arch_svg(spec["arch"], spec["key"])}'
        f'<div class="specs"><div><span class="k">관찰</span><span>{esc(spec["obs"])}</span></div>'
        f'<div><span class="k">행동</span><span>{esc(spec["act"])}</span></div>'
        f'<div><span class="k">정책</span><span>{esc(spec["policy"])}</span></div></div></div>'
        f'<div class="panel panel-pad"><div class="panel-title">핵심 결과</div><p class="result-box">{spec["result"]}</p></div></div>'
        f'<div class="sec"><h2 class="sec-title"><span class="sec-num">01</span>이 계보가 답하는 질문</h2>'
        f'<div class="panel panel-pad question prose">{f"<h3>{question}</h3>" if question else ""}<div class="stack">{prose}</div></div></div>'
        + (f'<div class="sec"><h2 class="sec-title"><span class="sec-num">02</span>이전 단계</h2><div class="stack">{roots}</div></div>' if roots else '')
        + f'<div class="sec"><h2 class="sec-title"><span class="sec-num">{"03" if roots else "02"}</span>실험 한눈에</h2>{listing}</div>'
        + render_pn(model, sequence, 'overview') + '</section>')


def render_run_overview(model, page, sequence, members):
    hero = page['hero'] or []
    title = page_title(model, page['id'])
    body = ''.join(render_block(b) for b in hero if b['type'] not in ('h1', 'h2'))
    roots = ''.join(f'<div class="callout"><span class="callout-ico">↺</span><div class="callout-body">{r}</div></div>' for r in page['root_rows'])
    extra = render_sections(page['blocks'])
    sessions = [m for m in members if m != page['id']]
    info = model['info'].get(page['id'], {})
    return (f'<section class="page" data-page="{esc(page["id"])}" data-title="{esc(title)}"><div class="page-head"><span class="eyebrow">run 개요</span>'
            f'<h1>{esc(title)}</h1><div class="meta"><span class="pill s-{page_status(model, page["id"])}">{esc(info.get("tag") or "run")}</span>'
            f'<span class="chip">세션 {len(sessions)}개</span></div></div><div class="stack">{body}{roots}</div>{extra}'
            + (f'<div class="sec"><h2 class="sec-title"><span class="sec-num">→</span>세션 순서</h2>{exp_cards(model, sessions)}</div>' if sessions else
               '<div class="sec"><div class="callout warn"><span class="callout-ico">!</span><div class="callout-body">이 run에는 실행된 세션 기록이 없습니다.</div></div></div>')
            + render_pn(model, sequence, page['id']) + '</section>')


def render_experiment(model, page, sequence, group_title, members):
    card = page['card'] or dict(title=page['id'], steps='', status='', cls='base')
    number, name = split_title(card['title'])
    position = f' · {members.index(page["id"]) + 1}/{len(members)}' if page['id'] in members else ''
    parents = [a for a, b in model['edges'] if b == page['id']]
    children = [b for a, b in model['edges'] if a == page['id']]

    def chips(ids):
        return ''.join(f'<a class="chip" href="#{esc(i)}"><span class="dot s-{page_status(model, i)}"></span>{esc(page_title(model, i))}</a>' for i in ids)

    lineage = ''
    if parents or children:
        lineage = ('<div class="lineage">' + (f'<span class="lbl">부모</span>{chips(parents)}' if parents else '')
                   + (f'<span class="lbl" style="margin-left:8px">자식</span>{chips(children)}' if children else '') + '</div>')
    context = f'<div class="context">{page["context"]}</div>' if page['context'] else ''
    steps = f'<span class="chip mono">{esc(card["steps"])}</span>' if card['steps'] else ''
    return (f'<section class="page" data-page="{esc(page["id"])}" data-title="{esc(card["title"])}"><div class="page-head">'
            f'<span class="eyebrow">{esc(group_title)}{position}</span>'
            f'<h1>{f"<span class=t-num>{esc(number)}</span>" if number else ""}{esc(name)}</h1>'
            f'<div class="meta"><span class="pill s-{card["cls"]}">{esc(card["status"] or "기록")}</span>{steps}</div>{lineage}{context}</div>'
            f'{render_sections(page["blocks"])}{render_pn(model, sequence, page["id"])}</section>')


def topbar(active_key, inside_pages):
    prefix = '' if inside_pages else 'pages/'
    home = '../index.html' if inside_pages else 'index.html'
    switch = ''.join(f'<a class="{"on" if a["key"] == active_key else ""}" href="{prefix}{a["key"]}.html">{esc(a["short"])}</a>' for a in APPROACHES)
    current = next((a for a in APPROACHES if a['key'] == active_key), None)
    crumbs = ''
    if current:
        crumbs = (f'<nav class="crumbs" aria-label="현재 위치"><a class="hide-sm" href="{home}">갤러리</a><span class="sep hide-sm">/</span>'
                  f'<a class="hide-sm" href="#overview">{esc(current["name"])}</a><span class="sep hide-sm">/</span><b data-crumb>개요</b></nav>')
    menu = f'<button class="icon-btn menu-btn" type="button" data-menu-toggle aria-label="실험 목록 열기">{ICON_MENU}</button>' if current else ''
    return (f'<header class="topbar"><div class="topbar-in">{menu}<a class="brand" href="{home}"><span class="brand-mark"></span>Space Z <b>RL 아카이브</b></a>'
            f'{crumbs}<span class="spacer"></span><nav class="switch" aria-label="계열 전환">{switch}</nav>'
            f'<button class="icon-btn" type="button" data-theme-toggle aria-label="밝기 테마 전환">{ICON_MOON}{ICON_SUN}</button></div></header>')


def head(title, description, prefix):
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="description" content="{esc(description)}"><meta name="robots" content="noindex,nofollow"><title>{esc(title)}</title>'
            f'{THEME_BOOT}<link rel="stylesheet" href="{prefix}assets/site.css"><script src="{prefix}assets/site.js" defer></script></head>')


LIGHTBOX = (f'<div class="lightbox" hidden><button class="icon-btn" type="button" data-close aria-label="닫기">{ICON_CLOSE}</button>'
            f'<figure><img alt=""><figcaption></figcaption></figure></div>')


def render_approach(model):
    spec = model['spec']
    svg, flow_order, rows = flow_svg(model)
    groups, sequence = build_navigation(model, flow_order, rows)
    group_of = {m: (title, members) for title, members in groups for m in members}
    sections = [render_overview(model, groups, sequence)]
    for page_id in sequence[1:]:
        page = model['by_id'][page_id]
        title, members = group_of[page_id]
        if page['card'] is None and page['hero'] is not None:
            sections.append(render_run_overview(model, page, sequence, members))
        else:
            sections.append(render_experiment(model, page, sequence, title, members))
    legend = ''.join(f'<span><span class="dot s-{k}"></span>{v}</span>' for k, v in LEGEND
                     if any(i['cls'] == k for i in model['info'].values()))
    flow = (f'<section class="panel flow" aria-label="실험 계보"><div class="flow-head"><h2>계보</h2>'
            f'<span class="hint">왼쪽 → 오른쪽으로 이어 학습 · 노드를 누르면 해당 실험으로 이동</span><div class="legend">{legend}</div></div>'
            f'<div class="flow-scroll">{svg}</div></section>')
    document = (head(f'{spec["name"]} · Space Z RL 아카이브', spec['summary'], '../') +
                f'<body>{topbar(spec["key"], True)}<div class="shell">{render_sidebar(model, groups)}'
                f'<main class="main">{flow}{"".join(sections)}</main></div>{LIGHTBOX}</body></html>')
    return document, len(sequence) - 1


def render_index(models, counts):
    total_pages = sum(counts.values())
    total_shots = sum(len(m['ctx'].images) for m in models)
    cards = []
    for model in models:
        spec = model['spec']
        cards.append(
            f'<a class="panel acard" style="--tone:{spec["tone"]}" href="pages/{spec["key"]}.html"><span class="tag">{esc(spec["tag"])}</span>'
            f'<h2>{esc(spec["name"])}</h2><p class="desc">{esc(spec["summary"])}</p>'
            f'<div class="arch">{arch_svg(spec["arch"], "g-" + spec["key"])}</div>'
            f'<div class="specs"><div><span class="k">관찰</span><span>{esc(spec["obs"])}</span></div>'
            f'<div><span class="k">행동</span><span>{esc(spec["act"])}</span></div>'
            f'<div><span class="k">정책</span><span>{esc(spec["policy"])}</span></div></div>'
            f'<div class="result">{spec["result"]}</div>'
            f'<div class="foot"><span>{counts[spec["key"]]}개 페이지</span><span class="go">계보 보기 →</span></div></a>')
    rows = ''.join(f'<tr><td><a href="pages/{m["spec"]["key"]}.html">{esc(m["spec"]["name"])}</a></td><td>{esc(m["spec"]["obs"])}</td>'
                   f'<td>{esc(m["spec"]["act"])}</td><td>{esc(m["spec"]["policy"])}</td><td class="num">{counts[m["spec"]["key"]]}</td>'
                   f'<td>{esc(m["spec"]["headline"])}</td></tr>' for m in models)
    sample = chart_svg(dict(title='점수 분포 · 척도 0~400점', rows=[dict(label='예시', left=10, width=40, marks=dict(
        p25=dict(pos=10, text='p25 40'), p75=dict(pos=50, text='p75 200'), med=dict(pos=22, text='med 88'),
        mean=dict(pos=34, text='mean 136'), max=dict(pos=92, text='max 368')))], notes=[]))
    guide = (
        '<div class="guide">'
        '<div class="panel"><span class="eyebrow">01 · 계보</span><h3>무엇에서 갈라져 나왔나</h3><p>각 계열 페이지 상단의 그래프는 체크포인트를 이어받은 순서입니다. '
        '왼쪽에서 오른쪽으로 학습이 이어지고, 갈래가 나뉘면 같은 부모에서 변수 하나씩만 바꾼 형제 실험입니다. 색은 결과(개선·보류·실패)를 뜻합니다.</p></div>'
        f'<div class="panel"><span class="eyebrow">02 · 분포</span><h3>평균 하나로 판단하지 않기</h3>{sample}<p>상자는 p25–p75, 검은 선은 median, 빨간 마름모는 mean, '
        '초록 선은 단일 시드 최고값입니다. mean이 median보다 크게 오른쪽이면 소수 시드가 평균을 끌어올린 것입니다.</p></div>'
        '<div class="panel"><span class="eyebrow">03 · held-out</span><h3>고를 때 쓴 시드로 채점하지 않기</h3><p>체크포인트를 고를 때 쓴 dev 시드와 별개로, '
        '학습·선택 어디에도 쓰지 않은 새 시드로 한 번만 다시 확인합니다. dev에서 이겨도 held-out에서 지면 채택하지 않습니다.</p></div>'
        '</div>')
    return (head('Space Z RL 아카이브', '같은 게임을 네 가지 관찰·행동·정책 설계로 학습한 강화학습 실험 아카이브', '') +
            f'<body>{topbar(None, False)}<main class="home">'
            '<section class="home-hero"><span class="eyebrow">Space Z · 강화학습 실험 아카이브</span>'
            '<h1>같은 게임, 네 가지 학습 설계</h1>'
            '<p>같은 Space Z 게임 코어 위에서 관찰 구성·행동 공간·정책 구조를 다르게 잡은 네 계열의 PPO 실험입니다. '
            '계열을 고르면 실험마다 무엇을 바꿨고, 분포와 held-out 검증으로 어떤 결론이 났는지 처음부터 끝까지 따라갈 수 있습니다.</p>'
            f'<div class="home-stats"><div class="panel stat"><b>4</b><span>학습 설계</span></div>'
            f'<div class="panel stat"><b>{total_pages}</b><span>실험·세션 페이지</span></div>'
            f'<div class="panel stat"><b>{total_shots}</b><span>실제 플레이 화면</span></div></div></section>'
            f'<div class="cards">{"".join(cards)}</div>'
            '<section class="home-sec"><h2>한눈에 비교</h2><p>평가 조건(에피소드 길이·시드 수·점수 척도)이 계열마다 달라, 점수의 절대값은 계열끼리 직접 비교하지 않습니다.</p>'
            '<div class="tbl-wrap"><table class="tbl"><thead><tr><th>계열</th><th>관찰</th><th>행동</th><th>정책 구조</th><th class="num">페이지</th><th>핵심 결과</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></section>'
            f'<section class="home-sec"><h2>읽는 법</h2><p>모든 계열 페이지는 같은 구성으로 되어 있습니다.</p>{guide}</section>'
            '<div class="home-foot">수치는 각 실험 폴더의 config/experiment.json, episodes·evaluations 로그, confirmation 기록에서 가져온 조사 시점의 값입니다. '
            '자동 갱신되는 대시보드가 아니라 전달용 스냅샷입니다.</div>'
            f'</main>{LIGHTBOX}</body></html>')


# ---------------------------------------------------------------- output

def validate(out_dir):
    problems = []
    for page in sorted(out_dir.rglob('*.html')):
        soup = BeautifulSoup(page.read_text(encoding='utf-8'), 'html.parser')
        ids = {t['id'] for t in soup.find_all(id=True)} | {t['data-page'] for t in soup.find_all(attrs={'data-page': True})}
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            if href.startswith('#'):
                if href[1:] not in ids:
                    problems.append(f'{page.relative_to(out_dir)}: broken anchor {href}')
            elif not (page.parent / href.split('#')[0]).resolve().exists():
                problems.append(f'{page.relative_to(out_dir)}: broken link {href}')
        for tag in soup.find_all(['img', 'script', 'link']):
            ref = tag.get('src') or tag.get('href') or tag.get('data-full')
            if ref and not (page.parent / ref).resolve().exists():
                problems.append(f'{page.relative_to(out_dir)}: missing file {ref}')
        for button in soup.select('[data-full]'):
            if not (page.parent / button['data-full']).resolve().exists():
                problems.append(f'{page.relative_to(out_dir)}: missing image {button["data-full"]}')
    return problems


def empty_directory(path):
    """다른 프로세스가 폴더를 열어 두어도 지워지도록 내용만 비운다."""
    for child in path.iterdir():
        if child.is_dir():
            empty_directory(child)
            try:
                child.rmdir()
            except OSError:
                pass
        else:
            child.unlink()


def main():
    models = [parse_approach(spec) for spec in APPROACHES]
    OUT.mkdir(exist_ok=True)
    empty_directory(OUT)
    (OUT / 'pages').mkdir(exist_ok=True)
    (OUT / 'assets').mkdir(exist_ok=True)
    shutil.copy2(SRC / 'site.css', OUT / 'assets' / 'site.css')
    shutil.copy2(SRC / 'site.js', OUT / 'assets' / 'site.js')
    counts = {}
    for model in models:
        document, counts[model['spec']['key']] = render_approach(model)
        (OUT / 'pages' / f'{model["spec"]["key"]}.html').write_text(document, encoding='utf-8')
        for source, target in model['ctx'].images.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for warning in model['ctx'].warnings:
            print(f'[{model["spec"]["key"]}] {warning}')
    (OUT / 'index.html').write_text(render_index(models, counts), encoding='utf-8')
    problems = validate(OUT)
    for problem in problems:
        print('PROBLEM', problem)
    archive = shutil.make_archive(str(ZIP_BASE), 'zip', OUT)
    print(f'pages: {counts} · images: {sum(len(m["ctx"].images) for m in models)} · problems: {len(problems)} · zip: {archive}')


if __name__ == '__main__':
    main()
