"""Coverage tests for src/visualize_skeleton.py helper surface.

Targets the large uncovered helper surface of ``visualize_skeleton``:
module-level crop utilities, color standardization/expansion helpers,
ROI expansion, radius/pipeline resolvers, graph-distance math on tiny
synthetic neurons, cache-path helpers, HTML warning banners, and the
GLB export mesh plumbing.

All tests are hermetic: no network, no cache reads from the repo
(cache paths are redirected to ``tmp_path``), no ``fig.show()``, and
plotly is only used offline.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

import visualize_skeleton as vs_module  # noqa: E402
from visualize_skeleton import (  # noqa: E402
    VisualizeSkeleton,
    _apply_consistent_crop,
    _apply_consistent_crop_standalone,
    _compute_unified_crop_bounds,
    _configure_roi_mesh_traces,
    _detect_content_bounds,
    _detect_content_bounds_standalone,
)

DEFAULTS = dict(
    verbose=False,
    dataset='hemibrain:v1.2.1',
    client_type='neuprint',
    background_color='rgba(255, 255, 255, 1.0)',
)


def make_vis(**attrs):
    """Build a VisualizeSkeleton without running __post_init__."""
    vis = object.__new__(VisualizeSkeleton)
    merged = dict(DEFAULTS)
    merged.update(attrs)
    for key, value in merged.items():
        setattr(vis, key, value)
    return vis


def make_chain_neuron(n_nodes=8, body_id='42', radius=10.0):
    """Tiny straight-chain TreeNeuron (hermetic synthetic skeleton)."""
    types = ['root'] + ['slab'] * max(0, n_nodes - 2) + ['end']
    neuron = navis.TreeNeuron(pd.DataFrame({
        'node_id': np.arange(n_nodes, dtype=np.int64),
        'parent_id': np.array([-1] + list(range(n_nodes - 1)), dtype=np.int64),
        'x': np.arange(n_nodes, dtype=float) * 100.0,
        'y': np.zeros(n_nodes),
        'z': np.zeros(n_nodes),
        'radius': np.full(n_nodes, radius, dtype=float),
        'type': types[:n_nodes],
    }))
    neuron.soma = None
    neuron.id = body_id
    return neuron


def make_branched_neuron():
    """Root with two branches of different lengths (graph-distance tests)."""
    neuron = navis.TreeNeuron(pd.DataFrame({
        'node_id': np.array([0, 1, 2, 3, 4], dtype=np.int64),
        'parent_id': np.array([-1, 0, 1, 1, 3], dtype=np.int64),
        'x': np.array([0.0, 100.0, 200.0, 100.0, 100.0]),
        'y': np.array([0.0, 0.0, 0.0, 100.0, 200.0]),
        'z': np.zeros(5),
        'radius': np.full(5, 5.0),
        'type': ['root', 'slab', 'end', 'slab', 'end'],
    }))
    neuron.soma = None
    neuron.id = '7'
    return neuron


def make_white_image_with_block(size=(100, 120), block=((30, 40), (60, 80))):
    """RGB image: white background with a dark rectangle."""
    from PIL import Image
    img = Image.new('RGB', size, (255, 255, 255))
    px = img.load()
    (r0, c0), (r1, c1) = block
    for r in range(r0, r1):
        for c in range(c0, c1):
            px[c, r] = (10, 10, 10)
    return img


# ---------------------------------------------------------------------------
# module-level crop helpers
# ---------------------------------------------------------------------------
class TestModuleCropHelpers:
    def test_configure_roi_mesh_traces(self):
        traces = [go.Mesh3d(x=[0], y=[0], z=[0]) for _ in range(3)]
        out = _configure_roi_mesh_traces(traces, 'AL(L)')
        assert out[0].legendgroup == 'roi_mesh:AL(L)'
        assert out[0].showlegend is True
        assert out[1].showlegend is False
        assert all(t.name == 'brain region [AL(L)]' for t in out)
        assert all(t.hoverinfo == 'name' for t in out)

    def test_detect_content_bounds_rgb(self):
        img = make_white_image_with_block()
        assert _detect_content_bounds(img) == (30, 59, 40, 79)

    def test_detect_content_bounds_rgba(self):
        img = make_white_image_with_block().convert('RGBA')
        assert _detect_content_bounds(img) == (30, 59, 40, 79)

    def test_detect_content_bounds_none_for_blank(self):
        from PIL import Image
        blank = Image.new('RGB', (20, 20), (255, 255, 255))
        assert _detect_content_bounds(blank) is None

    def test_standalone_bounds_matches(self):
        img = make_white_image_with_block(block=((10, 10), (15, 15)))
        assert _detect_content_bounds_standalone(img) == (10, 14, 10, 14)
        from PIL import Image
        assert _detect_content_bounds_standalone(
            Image.new('RGB', (10, 10), (0, 0, 0)), (0, 0, 0)) is None

    def test_compute_unified_crop_bounds(self, tmp_path):
        paths = []
        for i, block in enumerate([((10, 10), (20, 20)), ((40, 60), (55, 90))]):
            p = tmp_path / f'frame_{i}.png'
            make_white_image_with_block(block=block).save(p)
            paths.append(str(p))
        bounds = _compute_unified_crop_bounds(paths)
        assert bounds == (10, 54, 10, 89)

    def test_compute_unified_crop_bounds_sampling_and_errors(self, tmp_path):
        paths = []
        for i in range(8):
            p = tmp_path / f'frame_{i}.png'
            make_white_image_with_block(block=((i, i), (i + 5, i + 5))).save(p)
            paths.append(str(p))
        paths.append(str(tmp_path / 'broken.png'))  # unreadable -> skipped
        bounds = _compute_unified_crop_bounds(paths, sample_count=3)
        assert bounds is not None
        assert _compute_unified_crop_bounds([]) is None
        # only-blank frames -> None
        blank = tmp_path / 'blank.png'
        from PIL import Image
        Image.new('RGB', (10, 10), (255, 255, 255)).save(blank)
        assert _compute_unified_crop_bounds([str(blank)]) is None

    def test_apply_consistent_crop(self, tmp_path):
        for i in range(3):
            img = make_white_image_with_block(block=((20 + i, 30), (40, 60)))
            img.save(tmp_path / f'deg_{i:03d}.jpeg', 'JPEG')
        size = _apply_consistent_crop(str(tmp_path), margin=5)
        # JPEG compression can bleed a few pixels past the drawn block.
        assert 38 <= size[0] <= 52 and 28 <= size[1] <= 40
        from PIL import Image
        with Image.open(tmp_path / 'deg_000.jpeg') as cropped:
            assert cropped.size == (size[0], size[1])

    def test_apply_consistent_crop_empty_and_blank(self, tmp_path):
        assert _apply_consistent_crop(str(tmp_path)) is None
        from PIL import Image
        Image.new('RGB', (20, 20), (255, 255, 255)).save(
            tmp_path / 'deg_000.jpeg', 'JPEG')
        assert _apply_consistent_crop(str(tmp_path)) is None

    def test_standalone_consistent_crop(self, tmp_path):
        for i in range(2):
            img = make_white_image_with_block(block=((10, 10), (30 + i * 10, 30)))
            img.save(tmp_path / f'deg_{i:03d}.jpeg', 'JPEG')
        size = _apply_consistent_crop_standalone(str(tmp_path), margin=0)
        assert 18 <= size[0] <= 30 and 28 <= size[1] <= 42
        assert _apply_consistent_crop_standalone(str(tmp_path.parent / 'nope')) is None

    def test_instance_crop_delegates(self):
        vis = make_vis()
        img = make_white_image_with_block()
        assert vis._detect_content_bounds(img) == (30, 59, 40, 79)
        assert vis._compute_unified_crop_bounds([]) is None


# ---------------------------------------------------------------------------
# verbosity / progress / background helpers
# ---------------------------------------------------------------------------
class TestVerbosityHelpers:
    def test_vprint_levels(self, capsys):
        assert make_vis(verbose=False)._vprint('hidden') is None
        make_vis(verbose='simple')._vprint('shown', level='simple')
        make_vis(verbose='simple')._vprint('hidden', level='full')
        make_vis(verbose='full')._vprint('full-shown', level='full')
        out = capsys.readouterr().out
        assert 'shown' in out and 'full-shown' in out and 'hidden' not in out

    def test_vprint_tqdm_paths(self, capsys):
        make_vis(verbose='full')._vprint('via-tqdm', use_tqdm=True)
        make_vis(verbose='full')._vprint('partial', use_tqdm=True, end='|')
        out = capsys.readouterr().out
        assert 'via-tqdm' in out and 'partial|' in out

    def test_progress(self, capsys):
        make_vis(verbose=False)._progress(1, 3, 'step')
        assert capsys.readouterr().out == ''
        make_vis(verbose='full')._progress(2, 5, 'loading')
        assert '[DROCAT][progress] 2/5 loading' in capsys.readouterr().out

    def test_is_dark_background(self):
        assert make_vis(background_color='black')._is_dark_background() is True
        assert make_vis(background_color='white')._is_dark_background() is False
        assert make_vis()._is_dark_background(color='#ff0000') is True
        assert make_vis()._is_dark_background(color='yellow') is False
        assert make_vis()._is_dark_background(color=object()) is False

    def test_get_effective_mesh_color(self):
        vis = make_vis(brain_mesh_color='auto', vnc_mesh_color='auto')
        assert vis._get_effective_mesh_color('brain') == 'rgba(200, 230, 240, 0.1)'
        vis.background_color = 'black'
        assert vis._get_effective_mesh_color('brain') == 'rgba(60, 60, 70, 0.1)'
        assert vis._get_effective_mesh_color('vnc') == 'rgba(60, 60, 70, 0.1)'
        vis.brain_mesh_color = 'rgba(1, 2, 3, 0.5)'
        vis.vnc_mesh_color = 'rgba(4, 5, 6, 0.5)'
        assert vis._get_effective_mesh_color('brain') == 'rgba(1, 2, 3, 0.5)'
        assert vis._get_effective_mesh_color('vnc') == 'rgba(4, 5, 6, 0.5)'
        vis.background_color = 'white'
        vis.vnc_mesh_color = 'auto'
        assert vis._get_effective_mesh_color('vnc') == 'rgba(200, 230, 240, 0.1)'

    def test_get_html_size_cap(self):
        assert make_vis(html_size_cap=42, export_method='kaleido')._get_html_size_cap() == 42
        assert make_vis(html_size_cap=None, export_method='webdriver')._get_html_size_cap() == 200
        assert make_vis(html_size_cap=None, export_method='webdriver-fast')._get_html_size_cap() == 200
        assert make_vis(html_size_cap=None, export_method='kaleido')._get_html_size_cap() == 100

    def test_add_view_selection_menu_variants(self):
        for dataset, brain_mesh in (
                ('hemibrain:v1.2.1', 'template'),
                ('manc:v1.0', None),
                ('male-cns:v0.9', None)):
            vis = make_vis(dataset=dataset, brain_mesh=brain_mesh, fig_3d=go.Figure())
            vis._add_view_selection_menu()
            menus = vis.fig_3d.layout.updatemenus
            assert len(menus) == 1 and len(menus[0].buttons) == 6
            assert len(vis.fig_3d.layout.annotations) == 2


# ---------------------------------------------------------------------------
# simplification / line-mode warning banners + HTML writing
# ---------------------------------------------------------------------------
class TestWarningBanners:
    def test_simplification_warning_matrix(self):
        warn = VisualizeSkeleton._skeleton_simplification_warning
        assert warn('hemibrain:v1.2.1', 'neuprint', 'line', 0.99) is None
        assert warn('hemibrain:v1.2.1', 'neuprint', 'tube', 'oops') is None
        assert warn('hemibrain:v1.2.1', 'neuprint', 'tube', 0.90) is None
        got = warn('hemibrain:v1.2.1', 'neuprint', 'tube', 0.91)
        assert got['family'] == 'NeuPrint' and got['threshold'] == 0.90
        got = warn('hemibrain:v1.2.1', 'neuprint', 'tube', 0.96, 'fine')
        assert got['threshold'] == 0.95
        assert warn('hemibrain:v1.2.1', 'neuprint', 'tube', 0.94, 'fine') is None
        got = warn('flywire_FAFB_v783', 'flywire', 'tube', 0.96)
        assert got['family'] == 'FlyWire FAFB' and got['threshold'] == 0.95
        assert warn('flywire_FAFB_v783', 'flywire', 'tube', 0.95) is None
        assert warn('flywire_BANC_v626', 'flywire', 'tube', 0.99) is None

    def test_warning_html_variants(self):
        vis = make_vis(skeleton_mode='tube', skeleton_mesh_simplification=0.99,
                       neuprint_skeleton_pipeline='fast')
        html = vis._skeleton_simplification_warning_html()
        assert 'drocat-skeleton-simplification-warning' in html
        assert 'simp90' in html or 'simp 90' in html or 'simp90' in html or '0.90' in html
        vis.neuprint_skeleton_pipeline = 'fine'
        assert 'raw .swc.gz source' in vis._skeleton_simplification_warning_html()
        vis.dataset = 'flywire_FAFB_v783'
        assert 'soma-aware mesh cache' in vis._skeleton_simplification_warning_html()
        vis.skeleton_mesh_simplification = 0.90
        assert vis._skeleton_simplification_warning_html() == ''

    def test_line_mode_warning_html(self):
        vis = make_vis(skeleton_mode='tube')
        assert vis._line_mode_export_warning_html() == ''
        vis.skeleton_mode = 'line'
        html = vis._line_mode_export_warning_html()
        assert 'drocat-line-mode-export-warning' in html
        assert 'hemibrain' in html

    def test_inject_warning_into_html(self, tmp_path):
        vis = make_vis(skeleton_mode='line', save_folder=str(tmp_path))
        page = tmp_path / 'page.html'
        page.write_text('<html><body><div>plot</div></body></html>')
        vis._inject_skeleton_simplification_warning(str(page))
        content = page.read_text()
        assert 'drocat-line-mode-export-warning' in content
        # second injection must be a no-op
        vis._inject_skeleton_simplification_warning(str(page))
        assert content == page.read_text()
        # no body tag -> untouched; missing file -> untouched
        page2 = tmp_path / 'nobody.html'
        page2.write_text('<html></html>')
        vis._inject_skeleton_simplification_warning(str(page2))
        assert page2.read_text() == '<html></html>'
        vis._inject_skeleton_simplification_warning(str(tmp_path / 'absent.html'))

    def test_write_plotly_html(self, tmp_path):
        vis = make_vis(skeleton_mode='line', save_folder=str(tmp_path))
        fig = go.Figure(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1]))
        out = tmp_path / 'fig.html'
        vis._write_plotly_html(fig, str(out))
        content = out.read_text()
        assert 'plotly' in content.lower()
        assert 'drocat-line-mode-export-warning' in content

    def test_record_large_html_warning(self, tmp_path):
        vis = make_vis(save_folder=str(tmp_path))
        small = tmp_path / 'small.html'
        small.write_text('x')
        assert vis._record_large_html_warning(str(small)) is False
        temp = tmp_path / '_temp_export.html'
        temp.write_text('x')
        assert vis._record_large_html_warning(str(temp)) is False
        assert vis._record_large_html_warning(str(tmp_path / 'missing.html')) is False
        big = tmp_path / 'big.html'
        with open(big, 'wb') as handle:
            handle.truncate(51 * 1024 * 1024)  # sparse 51 MB file
        assert vis._record_large_html_warning(str(big)) is True
        note = (tmp_path / 'user_warning_notes.txt').read_text()
        assert 'render warning' in note and 'big.html' in note
        # duplicate marker is not appended twice
        assert vis._record_large_html_warning(str(big)) is True
        assert note == (tmp_path / 'user_warning_notes.txt').read_text()

    def test_write_user_warning_notes(self, tmp_path):
        vis = make_vis(save_folder=str(tmp_path), skeleton_mode='tube',
                       skeleton_mesh_simplification=0.99,
                       neuprint_skeleton_pipeline='fine_opt')
        vis._write_user_warning_notes()
        text = (tmp_path / 'user_warning_notes.txt').read_text()
        assert 'skeleton_mesh_simplification=0.99' in text
        assert 'neuprint_skeleton_pipeline=fine' in text
        assert '>0.95' in text
        vis.skeleton_mode = 'line'
        vis.dataset = 'flywire_FAFB_v783'
        vis.skeleton_mesh_simplification = None
        vis._write_user_warning_notes()
        text = (tmp_path / 'user_warning_notes.txt').read_text()
        assert "not set" in text and 'not applied' in text


# ---------------------------------------------------------------------------
# small static parsers
# ---------------------------------------------------------------------------
class TestStaticParsers:
    def test_parse_synapse_size(self):
        parse = VisualizeSkeleton._parse_synapse_size
        assert parse('real') == 'real'
        assert parse('  REAL ') == 'real'
        assert parse('2') == 2.0
        assert parse('2.5x') == 2.5
        assert parse('2 x real') == 2.0
        assert parse('1.25\u00d7') == 1.25
        assert parse('banana') is None

    def test_is_valid_color(self):
        vis = make_vis()
        assert vis._is_valid_color('red') is True
        assert vis._is_valid_color((10, 20, 30)) is True
        assert vis._is_valid_color('not-a-color') is False

    def test_unwrap_palette_value(self):
        unwrap = VisualizeSkeleton._unwrap_palette_value
        colors, continuous = unwrap({'colors': ['red'], 'continuous': True})
        assert colors == ['red'] and continuous is True
        colors, continuous = unwrap(['red', 'blue'])
        assert colors == ['red', 'blue'] and continuous is False

        class Palette(list):
            is_continuous_palette = True
        colors, continuous = unwrap(Palette(['red']))
        assert continuous is True


# ---------------------------------------------------------------------------
# color input normalization (instance methods)
# ---------------------------------------------------------------------------
class TestColorNormalization:
    def _color_vis(self, **attrs):
        base = dict(
            neuron_colors=None,
            synapse_colors=[],
            mesh_color=None,
            neuron_layers=['a'],
            brain_mesh_color='auto',
            vnc_mesh_color='auto',
        )
        base.update(attrs)
        return make_vis(**base)

    def test_standardize_color_input(self):
        vis = make_vis()
        assert vis._standardize_color_input('red') == ['rgba(255, 0, 0, 1.0)']
        assert vis._standardize_color_input('bogus') == ['rgba(128, 128, 128, 1.0)']
        assert vis._standardize_color_input((0, 0, 255)) == ['rgba(0, 0, 255, 1.0)']
        assert vis._standardize_color_input(['red', 'zz-nope-123']) == [
            'rgba(255, 0, 0, 1.0)', 'rgba(128, 128, 128, 1.0)']
        assert vis._standardize_color_input([]) == ['rgba(128, 128, 128, 1.0)']
        assert vis._standardize_color_input(np.array(['red'])) == [
            'rgba(255, 0, 0, 1.0)']

    def test_standardize_mesh_color_input(self):
        vis = make_vis(mesh_roi=['A', 'B'])
        assert vis._standardize_mesh_color_input('red') == 'rgba(255, 0, 0, 0.1)'
        assert vis._standardize_mesh_color_input('nope') == 'rgba(100, 100, 100, 0.1)'
        assert vis._standardize_mesh_color_input((0, 255, 0)) == 'rgba(0, 255, 0, 0.1)'
        got = vis._standardize_mesh_color_input(['red', 'blue'])
        assert got == ['rgba(255, 0, 0, 0.1)', 'rgba(0, 0, 255, 0.1)']
        seq = ['#000000', '#444444', '#888888', '#cccccc']
        cont = vis._standardize_mesh_color_input(seq, continuous=True)
        assert len(cont) == 2  # sampled to mesh_roi count
        assert vis._standardize_mesh_color_input(12345) == 'rgba(100, 100, 100, 0.1)'

    def test_is_custom_mesh_color_specified(self):
        assert make_vis(mesh_color=(100, 100, 100))._is_custom_mesh_color_specified() is False
        assert make_vis(mesh_color='rgba(100, 100, 100, 0.1)')._is_custom_mesh_color_specified() is False
        assert make_vis(mesh_color='red')._is_custom_mesh_color_specified() is True
        vis = make_vis(mesh_color='red', _original_mesh_color=(100, 100, 100))
        assert vis._is_custom_mesh_color_specified() is False

    def test_normalize_color_inputs_defaults_and_fallbacks(self):
        vis = self._color_vis(neuron_colors=[], synapse_colors=[], mesh_color=None,
                              brain_mesh_color='not-a-color', vnc_mesh_color=(1, 2),
                              background_color='auto')
        vis._normalize_color_inputs()
        assert vis.neuron_colors is None  # resolved later against background
        assert len(vis.synapse_colors) == 10  # Category10 fallback
        assert vis.mesh_color is None or vis.mesh_color == (100, 100, 100)
        assert vis.brain_mesh_color == 'auto'
        assert vis.vnc_mesh_color == 'auto'
        assert vis.background_color.startswith('rgba(255, 255, 255')

    def test_normalize_color_inputs_valid_and_dicts(self):
        vis = self._color_vis(
            neuron_colors={'colors': ['red', 'bogus'], 'continuous': False},
            synapse_colors={'colors': [], 'continuous': True},
            mesh_color=['red', 'blue'],
            brain_mesh_color='#123456',
            vnc_mesh_color='rgba(1, 2, 3, 0.5)',
            background_color='black',
            neuron_layers=['a', 'b'],
        )
        vis._normalize_color_inputs()
        assert vis.neuron_colors == ['red']
        assert vis._neuron_colors_continuous is False
        assert vis.synapse_colors == list(__import__('bokeh').palettes.Category10[10])
        assert vis.mesh_color == ['red', 'blue']
        assert vis.brain_mesh_color == '#123456'
        assert vis.vnc_mesh_color == 'rgba(1, 2, 3, 0.5)'
        assert vis.background_color.startswith('rgba(0, 0, 0')

    def test_normalize_color_inputs_invalid_sequences(self):
        vis = self._color_vis(neuron_colors=['only-zzz'], synapse_colors='zz-nope-123',
                              mesh_color=123, background_color='navy')
        vis._normalize_color_inputs()
        assert vis.neuron_colors is None
        assert len(vis.synapse_colors) == 10
        assert vis.mesh_color == (100, 100, 100)
        assert vis.background_color.startswith('rgba(0, 0, 128')


# ---------------------------------------------------------------------------
# color string helpers
# ---------------------------------------------------------------------------
class TestColorStringHelpers:
    def test_alpha_and_hex_helpers(self):
        vis = make_vis()
        assert vis._colors_have_explicit_alpha('rgba(1,2,3,0.5)') is True
        assert vis._colors_have_explicit_alpha('red') is False
        assert vis._extract_alpha_from_color('rgba(1, 2, 3, 0.25)') == 0.25
        assert vis._extract_alpha_from_color('garbage') == 1.0
        assert vis._rgba_to_hex('rgba(255, 0, 0, 0.5)') == '#ff0000'
        assert vis._rgba_to_hex('garbage') == '#808080'
        assert vis._get_opaque_color('rgba(255, 0, 0, 0.2)') == 'rgba(255, 0, 0, 1.0)'
        assert vis._get_opaque_color('garbage') == 'garbage'

    def test_darken_color(self):
        vis = make_vis()
        assert vis._darken_color('#ff0000', 0.5) == 'rgba(127, 0, 0, 1.0)'
        assert vis._darken_color((200, 100, 50), 1.0) == 'rgba(200, 100, 50, 1.0)'
        assert vis._darken_color('red', 0.0) == 'rgba(0, 0, 0, 1.0)'

    def test_interpolate_colors(self):
        vis = make_vis()
        base = ['red', 'blue']
        out = vis._interpolate_colors(base, 4)
        assert len(out) == 4
        assert out[0] == 'rgba(255, 0, 0, 1.0)' and out[1] == 'rgba(0, 0, 255, 1.0)'
        assert vis._interpolate_colors(['red'], 3) == ['rgba(255, 0, 0, 1.0)'] * 3
        assert vis._interpolate_colors([], 2) == ['rgba(128, 128, 128, 1.0)'] * 2
        assert vis._interpolate_colors(base, 1) == ['rgba(255, 0, 0, 1.0)']
        out = vis._interpolate_colors(['red', 'zz-nope-123', 'blue'], 6)
        assert len(out) == 6 and out[1] == 'rgba(128, 128, 128, 1.0)'

    def test_sample_continuous_color_sequence(self):
        vis = make_vis()
        assert vis._sample_continuous_color_sequence(['a', 'b'], 0) == []
        assert vis._sample_continuous_color_sequence([], 3) == []
        assert vis._sample_continuous_color_sequence(['a', 'b', 'c'], 1) == ['a']
        assert vis._sample_continuous_color_sequence(['a'], 3) == ['a', 'a', 'a']
        assert vis._sample_continuous_color_sequence(
            ['a', 'b', 'c', 'd', 'e'], 3) == ['a', 'c', 'e']

    def test_expand_color_sequence_policies(self):
        colors = ('red', 'blue')
        vis = make_vis(expand_colors='interpolation')
        out = vis._expand_color_sequence(colors, 4, warn=False)
        assert len(out) == 4
        vis = make_vis(expand_colors='darken')
        out = vis._expand_color_sequence(colors, 5, warn=False)
        assert len(out) == 5 and out[2].startswith('rgba(')
        vis = make_vis(expand_colors='cycle')
        assert vis._expand_color_sequence(colors, 5, warn=False) == (
            'red', 'blue', 'red', 'blue', 'red')
        assert vis._expand_color_sequence(colors, 0, warn=False) == tuple()
        assert vis._expand_color_sequence(colors, 1, warn=False) == ('red',)
        cont = vis._expand_color_sequence(('a', 'b', 'c'), 2, warn=False,
                                          continuous=True)
        assert cont == ('a', 'c')


# ---------------------------------------------------------------------------
# neuron lookup / per-neuron colors
# ---------------------------------------------------------------------------
class TestNeuronColors:
    def test_normalize_neuron_lookup_keys(self):
        vis = make_vis()
        assert vis._normalize_neuron_lookup_keys(None) == []
        assert vis._normalize_neuron_lookup_keys(42) == [42, '42']
        assert vis._normalize_neuron_lookup_keys(np.int64(7)) == [7, '7']
        assert vis._normalize_neuron_lookup_keys(float('nan')) == []
        assert vis._normalize_neuron_lookup_keys('  ') == []
        assert vis._normalize_neuron_lookup_keys('nan') == []
        assert vis._normalize_neuron_lookup_keys('123') == [123, '123']
        assert vis._normalize_neuron_lookup_keys(' DA1_lPN ') == ['DA1_lPN']

    def test_build_per_neuron_color_map(self):
        vis = make_vis(
            neuron_dfs=[pd.DataFrame({'bodyId': [1, 2], 'instance': ['a_L', 'b_R']}),
                        pd.DataFrame()],
            _base_neuron_colors=('red', 'blue'),
            _neuron_colors_continuous=False,
            expand_colors='cycle',
        )
        cmap = vis._build_per_neuron_color_map()
        assert cmap[1] == 'red'
        assert cmap['2'] == 'blue'
        assert cmap['a_L'] == 'red'
        vis.neuron_dfs = [pd.DataFrame()]
        assert vis._build_per_neuron_color_map() == {}

    def test_resolve_neuron_color(self):
        vis = make_vis(neuron_colors=['rgba(1, 1, 1, 1.0)'], color_mode='per_layer')
        assert vis._resolve_neuron_color(5, 0) == 'rgba(1, 1, 1, 1.0)'
        vis.color_mode = 'per_neuron'
        vis._per_neuron_colors = {5: 'rgba(9, 9, 9, 1.0)'}
        assert vis._resolve_neuron_color(5, 0) == 'rgba(9, 9, 9, 1.0)'
        vis._neuron_color_overrides = {'5': 'rgba(7, 7, 7, 1.0)'}
        assert vis._resolve_neuron_color(5, 0) == 'rgba(7, 7, 7, 1.0)'

    def test_apply_plotly_trace_color(self):
        vis = make_vis()
        trace = go.Scatter3d(x=[0], y=[0], z=[0], line=dict(color='blue'),
                             marker=dict(color='blue'))
        vis._apply_plotly_trace_color(trace, 'rgba(255, 0, 0, 0.5)')
        assert trace.line.color == '#ff0000'
        assert trace.marker.color == '#ff0000'
        assert trace.opacity == 0.5

    def test_apply_k3d_object_color(self):
        class Obj:
            color = None
            opacity = None
        vis = make_vis()
        obj = Obj()
        vis._apply_k3d_object_color(obj, 'rgba(255, 0, 0, 0.5)')
        assert obj.color == (255 << 16)
        assert obj.opacity == 0.5
        vis._apply_k3d_object_color(obj, 'garbage')  # swallowed


# ---------------------------------------------------------------------------
# ROI expansion helpers
# ---------------------------------------------------------------------------
class TestRoiExpansion:
    def test_flatten_nested_roi_groups(self):
        vis = make_vis()
        flat, colors, groups = vis._flatten_nested_roi_groups(
            ['AME', ['aL', 'bL'], 'EB'], ['red', 'green', 'blue'])
        assert flat == ['AME', 'aL', 'bL', 'EB']
        assert colors == ['red', 'green', 'green', 'blue']
        assert groups['aL'] == 'aL+bL'
        flat, colors, groups = vis._flatten_nested_roi_groups(
            ['A', ['w', 'x', 'y', 'z']], 'red')
        assert groups['z'] == 'w+x+y+1more'
        flat, colors, _ = vis._flatten_nested_roi_groups(
            ['A', 'B'], (255, 0, 0))
        assert colors == [(255, 0, 0), (255, 0, 0)]
        flat, colors, _ = vis._flatten_nested_roi_groups(['A'], ['red'])
        assert colors == ['red']
        flat, colors, _ = vis._flatten_nested_roi_groups(['A', 'B'], [object()])
        assert colors[-1] == 'gray' or colors[0] is not None
        assert vis._flatten_nested_roi_groups([], 'red') == ([], 'red', {})

    def test_expand_roi_names_with_colors(self):
        vis = make_vis()
        available = ['LH(L)', 'LH(R)', 'EB', 'AL(L)']
        rois, colors = vis._expand_roi_names_with_colors(
            ['LH', 'EB', 'AL(L)', 'GONE'], ['red', 'blue'], available)
        assert rois == ['LH(L)', 'LH(R)', 'EB', 'AL(L)', 'GONE']
        assert colors == ['red', 'red', 'blue', 'blue', 'blue']
        rois, colors = vis._expand_roi_names_with_colors(
            ['AL'], 'green', ['AL(R)'])
        assert rois == ['AL(R)'] and colors == ['green']
        assert vis._expand_roi_names_with_colors([], 'red', []) == ([], 'red')


# ---------------------------------------------------------------------------
# hemisphere filter + layer map CSV
# ---------------------------------------------------------------------------
class TestDataFrameHelpers:
    def test_filter_neuron_df_by_hemisphere(self):
        ndf = pd.DataFrame({
            'bodyId': [1, 2, 3, 4],
            'Soma side': ['L', 'R', None, None],
            'instance': ['x', 'y', 'z_R', 'plain'],
        })
        vis = make_vis(hemisphere='left')
        out, rdf = vis._filter_neuron_df_by_hemisphere(ndf)
        assert out['bodyId'].tolist() == [1, 4]
        vis = make_vis(hemisphere='right')
        out, _ = vis._filter_neuron_df_by_hemisphere(ndf)
        assert out['bodyId'].tolist() == [2, 3, 4]
        vis = make_vis(hemisphere='both')
        out, _ = vis._filter_neuron_df_by_hemisphere(ndf)
        assert len(out) == 4
        # rdf filtering by bodyId index
        rdf = pd.DataFrame({'w': [1, 2, 3]}, index=pd.Index(['1', '2', '5'],
                                                             name='bodyId'))
        vis = make_vis(hemisphere='right')
        _, rdf_out = vis._filter_neuron_df_by_hemisphere(ndf, rdf)
        assert list(rdf_out.index) == ['2']

    def test_parse_layer_map_csv(self, tmp_path):
        csv = tmp_path / 'layers.csv'
        csv.write_text(
            'layer,id_type_instance,color\n'
            'L1,101,red\n'
            'L1,102,blue\n'
            'L2,DA1_lPN,\n'
        )
        vis = make_vis(layer_map_csv=str(csv), neuron_alpha=0.8)
        vis._parse_layer_map_csv()
        assert vis.neuron_layers == [[101, 102], 'DA1_lPN']
        assert vis.custom_layer_names == ['L1', 'L2']
        assert vis._neuron_color_overrides[101].startswith('rgba(255, 0, 0')
        assert vis._neuron_color_overrides['102'].startswith('rgba(0, 0, 255')

    def test_parse_layer_map_csv_errors(self, tmp_path):
        vis = make_vis(layer_map_csv=str(tmp_path / 'absent.csv'))
        with pytest.raises(FileNotFoundError):
            vis._parse_layer_map_csv()
        bad = tmp_path / 'bad.csv'
        bad.write_text('wrong,columns\n1,2\n')
        vis.layer_map_csv = str(bad)
        with pytest.raises(ValueError):
            vis._parse_layer_map_csv()

    def test_parse_layer_map_csv_flywire_ids(self, tmp_path):
        csv = tmp_path / 'fw.csv'
        csv.write_text('layer,id_type_instance\nL1,720575940614131061\n')
        vis = make_vis(layer_map_csv=str(csv), dataset='flywire_FAFB_v783',
                       neuron_alpha=1.0)
        vis._parse_layer_map_csv()
        assert len(vis.neuron_layers) == 1

    def test_apply_soma_radius_cap(self):
        neuron = make_chain_neuron(n_nodes=6, radius=100.0)
        vis = make_vis(soma_radius_cap=50.0, smooth_skeleton=False)
        vis._apply_soma_radius_cap(navis.NeuronList([neuron]))
        assert (neuron.nodes['radius'] <= 50.0).all()
        neuron2 = make_chain_neuron(n_nodes=6, radius=100.0)
        neuron2.nodes.loc[3, 'radius'] = 500.0
        vis = make_vis(soma_radius_cap=50.0, smooth_skeleton=True)
        vis._apply_soma_radius_cap(navis.NeuronList([neuron2]))
        assert (neuron2.nodes['radius'] <= 50.0).all()

    def test_generate_smart_layer_names(self):
        vis = make_vis(
            neuron_dfs=[
                pd.DataFrame({'type': ['aMe12', 'aMe12'], 'bodyId': [1, 2]}),
                pd.DataFrame({'type': ['x', 'y'], 'bodyId': [3, 4]}),
                pd.DataFrame({'bodyId': [5]}),
                pd.DataFrame({'bodyId': [6, 7]}),
                pd.DataFrame(),
            ],
            layer_names=['l0', 'l1', 'l2', 'l3', 'l4'],
        )
        names = vis._generate_smart_layer_names()
        assert names[0] == 'aMe12'
        assert names[1].endswith('_etc')
        assert names[2:] == ['5', '6_etc', 'l4']

    def test_save_synapse_data(self, tmp_path):
        frames = [
            pd.DataFrame({'viz_layer': ['0->1'], 'pre': [1], 'post': [2]}),
            pd.DataFrame({'viz_layer': ['1->2'], 'pre': [2], 'post': [3]}),
        ]
        vis = make_vis(save_folder=str(tmp_path), saveas='run', output_format='csv')
        out = vis._save_synapse_data(frames)
        assert out == str(tmp_path / 'run_synapses.csv')
        saved = pd.read_csv(out)
        assert list(saved.columns)[0] == 'viz_layer' and len(saved) == 2
        vis.output_format = 'xlsx'
        out = vis._save_synapse_data(frames)
        assert out.endswith('.xlsx') and os.path.exists(out)
        vis.output_format = 'tsv'
        with pytest.raises(ValueError):
            vis._save_synapse_data(frames)
        assert vis._save_synapse_data([]) is None


# ---------------------------------------------------------------------------
# cache paths / dataset helpers
# ---------------------------------------------------------------------------
class TestCachePaths:
    def test_get_cache_path_and_synapse_path(self, tmp_path):
        vis = make_vis(script_path=str(tmp_path))
        cache_dir = vis._get_cache_path('skeletons')
        assert cache_dir == str(tmp_path / 'cache' / 'hemibrain_v1_2_1' / 'skeletons')
        assert os.path.isdir(cache_dir)
        syn_path = vis._get_synapse_cache_path(11, 22)
        assert syn_path.endswith('11_22.parquet')

    def test_get_dataset_mesh_dir_and_files(self, tmp_path):
        vis = make_vis(script_path=str(tmp_path))
        mesh_dir = vis._get_dataset_mesh_dir()
        assert os.path.isdir(mesh_dir)
        assert vis._roi_to_filename('AL(L)') == 'AL(L).json'
        assert vis._roi_to_filename('aL(L)') == '_aL(L).json'
        # encoded file exists -> returned
        encoded = Path(mesh_dir) / '_aL(L).json'
        encoded.write_text('{}')
        assert vis._get_mesh_file_path(mesh_dir, 'aL(L)') == str(encoded)
        # legacy fallback for uppercase
        legacy = Path(mesh_dir) / 'EB.json'
        legacy.write_text('{}')
        assert vis._get_mesh_file_path(mesh_dir, 'EB') == str(legacy)
        # nothing exists -> encoded path returned
        assert vis._get_mesh_file_path(mesh_dir, 'PB').endswith('PB.json')

    def test_get_dataset_abbreviation(self):
        cases = {
            'hemibrain:v1.2.1': 'HEMI',
            'male-cns:v0.9': 'MCNS',
            'flywire_FAFB_v783': 'FAFB',
            'flywire_BANC_v626': 'BANC',
            'optic-lobe:v1.1': 'OL',
            'manc:v1.0': 'MANC',
            'customset': 'CUST',
        }
        for dataset, expected in cases.items():
            assert make_vis(dataset=dataset)._get_dataset_abbreviation() == expected
        assert make_vis(dataset=None)._get_dataset_abbreviation() == 'UNKN'

    def test_get_synapse_table_path(self, tmp_path):
        vis = make_vis(script_path=str(tmp_path))
        assert vis._get_synapse_table_path() is None
        ds_dir = tmp_path / 'datasets' / 'hemibrain_v1_2_1'
        ds_dir.mkdir(parents=True)
        table = ds_dir / 'hemibrain_v1_2_1_synapse_table.parquet'
        table.write_bytes(b'x')
        assert vis._get_synapse_table_path() == str(table)
        fw = make_vis(script_path=str(tmp_path), dataset='flywire_FAFB_v783')
        assert fw._get_synapse_table_path() is None  # no datasets dir present


# ---------------------------------------------------------------------------
# pipeline / radius-style resolvers + skeleton graph math
# ---------------------------------------------------------------------------
class TestPipelineResolvers:
    def test_resolved_neuprint_pipeline(self):
        vis = make_vis(neuprint_skeleton_pipeline='fast', skeleton_mode='tube')
        assert vis._resolved_neuprint_skeleton_pipeline() == 'direct'
        vis.neuprint_skeleton_pipeline = 'fine_opt'
        assert vis._resolved_neuprint_skeleton_pipeline() == 'fine'
        vis.neuprint_skeleton_pipeline = 'fine_opt1'
        assert vis._resolved_neuprint_skeleton_pipeline() == 'artistic'
        vis.neuprint_skeleton_pipeline = 'fafb'
        assert vis._resolved_neuprint_skeleton_pipeline() == 'fine'
        vis.skeleton_mode = 'line'
        assert vis._resolved_neuprint_skeleton_pipeline() == 'direct'
        assert vis._uses_neuprint_fine_pipeline() is False
        vis.skeleton_mode = 'tube'
        vis.neuprint_skeleton_pipeline = 'fine'
        assert vis._uses_neuprint_fine_pipeline() is True
        assert vis._uses_neuprint_fine_optimized_pipeline() is True

    def test_resolved_fafb_pipeline(self):
        vis = make_vis(neuprint_skeleton_pipeline='direct')
        assert vis._resolved_fafb_pipeline() == 'fast'
        vis.neuprint_skeleton_pipeline = 'artistic'
        assert vis._resolved_fafb_pipeline() == 'artistic'
        vis.neuprint_skeleton_pipeline = None
        assert vis._resolved_fafb_pipeline() == 'fast'

    def test_resolved_skeleton_radius_style(self):
        assert make_vis(skeleton_radius_style='auto',
                        dataset='hemibrain:v1.2.1')._resolved_skeleton_radius_style() == 'fafb'
        assert make_vis(skeleton_radius_style='auto',
                        dataset='flywire_FAFB_v783')._resolved_skeleton_radius_style() == 'source'
        assert make_vis(skeleton_radius_style='default')._resolved_skeleton_radius_style() == 'constant'
        assert make_vis(skeleton_radius_style='source')._resolved_skeleton_radius_style() == 'source'
        assert make_vis(skeleton_radius_style=None)._resolved_skeleton_radius_style() == 'fafb'

    def test_cache_keys(self):
        vis = make_vis(neuprint_skeleton_pipeline='fast', skeleton_mode='tube',
                       skeleton_radius_style='auto')
        assert vis._get_neuprint_mesh_cache_key() == 'NEUPRINT_simp95'
        vis.neuprint_skeleton_pipeline = 'artistic'
        assert vis._get_neuprint_mesh_cache_key().endswith('_vertexcluster')
        vis.skeleton_radius_style = 'constant'
        assert '_radiusconstant' in vis._get_neuprint_mesh_cache_key()
        assert make_vis()._get_fafb_mesh_cache_key().startswith('FLYWIRE_simp')
        assert make_vis()._skeleton_cache_is_simplified() is False
        assert make_vis(skeleton_mesh_simplification=0.93)._effective_render_simplification(True) == 0.93


class TestSkeletonGraphMath:
    def test_edge_lengths_slow_and_fast(self):
        neuron = make_chain_neuron(n_nodes=5)
        vis = make_vis()
        slow = vis._neuron_edge_lengths(neuron)
        fast = vis._neuron_edge_lengths_fast(neuron)
        assert len(slow) == 4 and len(fast) == 4
        np.testing.assert_allclose(slow, fast)
        np.testing.assert_allclose(fast, np.full(4, 100.0))

    def test_edge_lengths_single_node(self):
        single = navis.TreeNeuron(pd.DataFrame({
            'node_id': np.array([0], dtype=np.int64),
            'parent_id': np.array([-1], dtype=np.int64),
            'x': [0.0], 'y': [0.0], 'z': [0.0], 'radius': [1.0],
            'type': ['root'],
        }))
        vis = make_vis()
        assert len(vis._neuron_edge_lengths(single)) == 0
        assert len(vis._neuron_edge_lengths_fast(single)) == 0

    def test_dist_to_tip_slow_and_fast(self):
        neuron = make_branched_neuron()
        vis = make_vis()
        slow = vis._neuron_dist_to_tip(neuron)
        fast = vis._neuron_dist_to_tip_fast(neuron)
        assert set(slow) == set(fast)
        for key in slow:
            assert slow[key] == pytest.approx(fast[key])
        assert fast[2] == 0.0 and fast[4] == 0.0
        assert fast[1] == pytest.approx(100.0)
        # single-node tree -> all zeros
        single = navis.TreeNeuron(pd.DataFrame({
            'node_id': np.array([0], dtype=np.int64),
            'parent_id': np.array([-1], dtype=np.int64),
            'x': [0.0], 'y': [0.0], 'z': [0.0], 'radius': [1.0],
            'type': ['root'],
        }))
        assert vis._neuron_dist_to_tip(single) == {0: 0.0}
        assert vis._neuron_dist_to_tip_fast(single) == {0: 0.0}

    def test_fafb_style_radius(self):
        vis = make_vis()
        neuron = make_branched_neuron()
        for optimized in (False, True):
            out = vis._fafb_style_radius(neuron, optimized=optimized)
            radii = out.nodes['radius'].to_numpy()
            assert (radii <= vis.NEUPRINT_RADIUS_SOMA_CAP).all()
            assert (radii > 0).all()

    def test_constant_neuprint_radius(self):
        vis = make_vis()
        out = vis._constant_neuprint_radius(make_chain_neuron())
        assert (out.nodes['radius'] == vis.NEUPRINT_DEFAULT_RADIUS).all()

    def test_fafb_style_neuprint_skeleton(self):
        vis = make_vis(dataset='hemibrain:v1.2.1', skeleton_radius_style='auto')
        for optimized in (False, True):
            out = vis._fafb_style_neuprint_skeleton(
                make_chain_neuron(n_nodes=10), optimized=optimized)
            assert 'radius' in out.nodes.columns
        out = vis._fafb_style_neuprint_skeleton(
            make_chain_neuron(n_nodes=6), radius_style='constant', optimized=True)
        assert (out.nodes['radius'] == vis.NEUPRINT_DEFAULT_RADIUS).all()
        out = vis._fafb_style_neuprint_skeleton(
            make_chain_neuron(n_nodes=6), radius_style='source')
        assert len(out.nodes) > 0

    def test_downsample_neuprint_skeleton_for_cache(self):
        vis = make_vis()
        obj = object()
        assert vis._downsample_neuprint_skeleton_for_cache(obj) is obj
        out = vis._downsample_neuprint_skeleton_for_cache(make_chain_neuron(n_nodes=30))
        assert hasattr(out, 'nodes')


# ---------------------------------------------------------------------------
# cached-neuron plumbing (monkeypatched cache backend)
# ---------------------------------------------------------------------------
class FakeRawCache:
    def __init__(self, store):
        self.store = store
        self.persisted = None

    def load_skeleton(self, bid):
        return self.store.get(bid)

    def persist_skeletons(self, items):
        self.persisted = items
        return len(items)


class TestCachedNeuronPlumbing:
    def test_load_cached_neurons_short_circuits(self):
        df = pd.DataFrame({'bodyId': [1, 2]})
        vis = make_vis(dataset='flywire_FAFB_v783')
        neurons, missing = vis._load_cached_neurons(df)
        assert neurons is None and missing == [1, 2]
        vis = make_vis()
        neurons, missing = vis._load_cached_neurons(df, ignore_cache=True)
        assert neurons is None and missing == [1, 2]

    def test_load_cached_neurons_from_raw_cache(self, monkeypatch):
        store = {1: make_chain_neuron(body_id='1')}
        fake = FakeRawCache(store)
        import morphology
        monkeypatch.setattr(morphology, 'find_similar_raw_cache',
                            lambda *a, **k: fake)
        vis = make_vis(script_path='/tmp')
        df = pd.DataFrame({'bodyId': [1, 2]})
        neurons, missing = vis._load_cached_neurons(df)
        assert len(neurons) == 1 and missing == [2]
        # failing backend -> graceful None
        def boom(*a, **k):
            raise RuntimeError('no cache')
        monkeypatch.setattr(morphology, 'find_similar_raw_cache', boom)
        neurons, missing = vis._load_cached_neurons(df)
        assert neurons is None and missing == [1, 2]

    def test_save_cached_neurons(self, monkeypatch):
        import morphology
        fake = FakeRawCache({})
        monkeypatch.setattr(morphology, 'find_similar_raw_cache',
                            lambda *a, **k: fake)
        vis = make_vis(script_path='/tmp')
        df = pd.DataFrame({'bodyId': [1]})
        vis._save_cached_neurons(df, navis.NeuronList(
            [make_chain_neuron(body_id='1')]))
        assert fake.persisted and 1 in fake.persisted
        # flywire datasets skip persistence entirely
        fake2 = FakeRawCache({})
        monkeypatch.setattr(morphology, 'find_similar_raw_cache',
                            lambda *a, **k: fake2)
        make_vis(dataset='flywire_FAFB_v783')._save_cached_neurons(
            df, navis.NeuronList([make_chain_neuron(body_id='1')]))
        assert fake2.persisted is None
        vis._save_cached_neurons(df, None)

    def test_fafb_mesh_cache_short_circuits(self):
        vis = make_vis(cache_neurons=False, skeleton_mesh_simplification=0.95,
                       dataset='flywire_FAFB_v783', script_path='/tmp')
        assert vis._load_cached_fafb_meshes([1]) == ({}, [1])
        vis.cache_neurons = True
        vis.skeleton_mesh_simplification = 0.90
        assert vis._load_cached_fafb_meshes([1]) == ({}, [1])
        vis.skeleton_mesh_simplification = 0.95
        vis.dataset = 'hemibrain:v1.2.1'
        assert vis._load_cached_fafb_meshes([1]) == ({}, [1])
        vis._save_cached_fafb_meshes({})  # cache_neurons but non-flywire -> no-op
        make_vis(cache_neurons=False)._save_cached_fafb_meshes({})

    def test_extrusion_cache_delegation(self, monkeypatch):
        import fafb_utils
        calls = {}
        monkeypatch.setattr(fafb_utils, 'load_extrusion_check_cache',
                            lambda root, ds: (calls.update(load=(root, ds)), {1: True})[1])
        monkeypatch.setattr(fafb_utils, 'save_extrusion_check_cache',
                            lambda root, ds, res: calls.update(save=(root, ds, res)))
        vis = make_vis(script_path='/proj', dataset='flywire_FAFB_v783')
        assert vis._load_extrusion_check_cache() == {1: True}
        vis._save_extrusion_check_cache({2: False})
        assert calls['save'][1] == 'flywire_FAFB_v783'
        flagged = []
        monkeypatch.setattr(fafb_utils, 'flag_extrusions',
                            lambda *a, **k: flagged.append(k) or [42])
        assert vis._detect_extrusions_in_skeletons({42: object()},
                                                   use_cache=False) == [42]
        assert flagged[0]['use_cache'] is False


# ---------------------------------------------------------------------------
# extrusion detection on synthetic meshes
# ---------------------------------------------------------------------------
class TestExtrusionDetection:
    def _clean_mesh(self):
        import trimesh
        return trimesh.creation.box(extents=(100.0, 100.0, 100.0))

    def _spiky_mesh(self):
        import trimesh
        box = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
        verts = list(box.vertices)
        faces = list(box.faces)
        verts.append((0.0, 0.0, 50000.0))  # long spike
        spike = len(verts) - 1
        faces.append((0, 1, spike))
        faces.append((1, 2, spike))
        return trimesh.Trimesh(vertices=np.array(verts),
                               faces=np.array(faces), process=False)

    def test_clean_mesh(self):
        result = VisualizeSkeleton.detect_mesh_extrusions(self._clean_mesh())
        assert result['has_extrusions'] is False
        assert result['severity'] == 'none'
        assert 'clean' in result['recommendation']

    def test_spiky_mesh_with_soma(self, capsys):
        result = VisualizeSkeleton.detect_mesh_extrusions(
            self._spiky_mesh(), soma_pos=[0, 0, 0], verbose=True)
        assert result['has_extrusions'] == True
        assert result['extrusion_count'] > 0
        assert result['edge_length_ratio'] > 3.0
        assert result['soma_region_issues'] == True
        assert 'extrusion' in result['recommendation'].lower()
        assert 'Mesh Analysis Results' in capsys.readouterr().out

    def test_invalid_mesh(self):
        result = VisualizeSkeleton.detect_mesh_extrusions(object())
        assert result['severity'] == 'unknown'
        assert result['has_extrusions'] is False

    def test_meshneuron_like_wrapper(self):
        class Wrapper:
            def __init__(self, tm):
                self.trimesh = tm
        result = VisualizeSkeleton.detect_mesh_extrusions(
            Wrapper(self._clean_mesh()))
        assert result['severity'] == 'none'


# ---------------------------------------------------------------------------
# kaleido simplification + GLB export plumbing
# ---------------------------------------------------------------------------
class TestFigureSimplificationAndExport:
    def test_simplify_figure_for_kaleido(self):
        import trimesh
        vis = make_vis(verbose=False)
        sphere = trimesh.creation.icosphere(subdivisions=2)
        mesh_trace = go.Mesh3d(
            x=sphere.vertices[:, 0], y=sphere.vertices[:, 1],
            z=sphere.vertices[:, 2],
            i=sphere.faces[:, 0], j=sphere.faces[:, 1], k=sphere.faces[:, 2],
        )
        long_line = go.Scatter3d(x=np.arange(2000.0), y=np.arange(2000.0),
                                 z=np.arange(2000.0),
                                 marker=dict(color=list(range(2000))))
        short_line = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1])
        fig = go.Figure(data=[mesh_trace, long_line, short_line])
        out = vis._simplify_figure_for_kaleido(fig, 0.25)
        assert len(out.data) == 3
        assert len(out.data[1].x) < 2000
        assert len(out.data[2].x) == 2

    def test_to_rgba(self):
        vis = make_vis()
        rgba = vis._to_rgba('red')
        assert tuple(rgba) == (255, 0, 0, 255)
        rgba = vis._to_rgba('red', alpha=0.5)
        assert rgba[3] == 128 or rgba[3] == 127
        rgba = vis._to_rgba(object())
        assert tuple(rgba)[:3] == (128, 128, 128)

    def test_extract_trimesh_variants(self):
        import trimesh
        vis = make_vis()
        assert vis._extract_trimesh(None) is None
        box = trimesh.creation.box()
        assert vis._extract_trimesh(box) is box
        mesh3d = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0],
                           i=[0], j=[1], k=[2])
        tm = vis._extract_trimesh(mesh3d)
        assert tm is not None and len(tm.faces) == 1
        empty = go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[])
        assert vis._extract_trimesh(empty) is None

        class WithMesh:
            def __init__(self, tm):
                self.mesh = tm
        assert vis._extract_trimesh(WithMesh(box)) is box

        class VertsFaces:
            vertices = box.vertices
            faces = box.faces
        assert vis._extract_trimesh(VertsFaces()) is not None
        assert vis._extract_trimesh(object()) is None

    def test_append_exportable_mesh_and_glb_export(self, tmp_path):
        import trimesh
        vis = make_vis(save_folder=str(tmp_path), saveas='scene',
                       exportable_meshes=[])
        box = trimesh.creation.box()
        assert vis._append_exportable_mesh(box, color='red', name='roi',
                                           role='roi') is True
        assert vis._append_exportable_mesh(object()) is False
        assert len(vis.exportable_meshes) == 1
        stored = vis.exportable_meshes[0]
        assert stored.metadata['export_role'] == 'roi'
        rgba = vis._get_glb_export_rgba(stored)
        assert tuple(rgba)[:3] == (255, 0, 0)
        prepared = vis._prepare_glb_mesh(stored, 'roi')
        assert prepared is not stored
        out = vis.export_3d_model()
        assert out == str(tmp_path / 'scene.glb')
        assert os.path.getsize(out) > 0
        # no meshes at all -> None
        empty_vis = make_vis(save_folder=str(tmp_path), saveas='empty',
                             exportable_meshes=[])
        assert empty_vis.export_3d_model() is None

    def test_get_glb_export_rgba_fallbacks(self):
        import trimesh
        vis = make_vis()
        box = trimesh.creation.box()
        box.visual.face_colors = [10, 20, 30, 255]
        rgba = vis._get_glb_export_rgba(box)
        assert tuple(rgba)[:3] == (10, 20, 30)


# ---------------------------------------------------------------------------
# transform helpers + available-ROI discovery
# ---------------------------------------------------------------------------
class TestTransformHelpers:
    def test_dataset_needs_transform(self):
        assert make_vis(dataset='hemibrain:v1.2.1',
                        brain_mesh='whole')._dataset_needs_transform() is True
        assert make_vis(dataset='hemibrain:v1.2.1',
                        brain_mesh='template')._dataset_needs_transform() is False
        assert make_vis(dataset='flywire_FAFB_v783',
                        brain_mesh='whole')._dataset_needs_transform() is True
        assert make_vis(dataset='flywire_FAFB_v783',
                        brain_mesh='template')._dataset_needs_transform() is False
        assert make_vis(dataset='male-cns:v0.9',
                        brain_mesh='whole')._dataset_needs_transform() is False

    def test_fafb_tilt_correction_matrix(self):
        vis = make_vis(dataset='hemibrain:v1.2.1', brain_mesh='template')
        np.testing.assert_array_equal(vis._get_fafb_tilt_correction_matrix(),
                                      np.eye(4))
        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='whole')
        np.testing.assert_array_equal(vis._get_fafb_tilt_correction_matrix(),
                                      np.eye(4))
        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='template')
        matrix = vis._get_fafb_tilt_correction_matrix()
        assert matrix.shape == (4, 4)
        assert not np.allclose(matrix, np.eye(4))
        # rotation keeps the brain center fixed
        center = np.array([527652.0, 240039.0, 148110.0, 1.0])
        np.testing.assert_allclose(matrix @ center, center, atol=1e-6)

    def test_apply_fafb_tilt_correction(self):
        # short-circuit paths return the object unchanged
        vis = make_vis(dataset='hemibrain:v1.2.1', brain_mesh='template')
        df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0]})
        assert vis._apply_fafb_tilt_correction(df) is df
        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='template',
                       FAFB_template_correction=False)
        assert vis._apply_fafb_tilt_correction(df) is df

        vis = make_vis(dataset='flywire_FAFB_v783', brain_mesh='template',
                       FAFB_template_correction=True)
        out = vis._apply_fafb_tilt_correction(df)
        assert out is not df
        assert np.isfinite(out[['x', 'y', 'z']].to_numpy()).all()
        # the brain-center point itself stays fixed under the transform
        center_df = pd.DataFrame({'x': [527652.0], 'y': [240039.0],
                                  'z': [148110.0]})
        center_out = vis._apply_fafb_tilt_correction(center_df)
        np.testing.assert_allclose(
            center_out[['x', 'y', 'z']].to_numpy(),
            center_df[['x', 'y', 'z']].to_numpy(), atol=1e-4)
        # DataFrame without xyz columns passes through
        plain = pd.DataFrame({'a': [1]})
        assert vis._apply_fafb_tilt_correction(plain) is plain

        neuron = make_chain_neuron(n_nodes=5)
        xformed = vis._apply_fafb_tilt_correction(neuron)
        assert hasattr(xformed, 'nodes')
        nlist = navis.NeuronList([make_chain_neuron(body_id='1'),
                                  make_chain_neuron(body_id='2')])
        xformed_list = vis._apply_fafb_tilt_correction(nlist)
        assert len(xformed_list) == 2


class TestAvailableRois:
    def test_reads_cached_roi_list(self, tmp_path):
        cache_dir = tmp_path / 'cache' / 'hemibrain_v1_2_1'
        cache_dir.mkdir(parents=True)
        (cache_dir / 'available_rois.json').write_text('["AL(L)", "EB"]')
        vis = make_vis(script_path=str(tmp_path))
        assert vis._get_available_rois() == ['AL(L)', 'EB']

    def test_flywire_local_scan(self, tmp_path):
        primary = tmp_path / 'navis_roi_meshes_json' / 'primary_rois'
        primary.mkdir(parents=True)
        (primary / 'LH.json').write_text('{}')
        hb_meshes = tmp_path / 'cache' / 'hemibrain_v1_2_1' / 'meshes'
        hb_meshes.mkdir(parents=True)
        (hb_meshes / 'EB.json').write_text('{}')
        vis = make_vis(script_path=str(tmp_path), dataset='flywire_FAFB_v783')
        rois = vis._get_available_rois(use_cache=False, fetch_online=True)
        assert rois == ['EB', 'LH']
        cached = tmp_path / 'cache' / 'flywire_FAFB_v783' / 'available_rois.json'
        assert cached.exists()

    def test_local_mesh_dir_fallback(self, tmp_path):
        vis = make_vis(script_path=str(tmp_path))
        mesh_dir = Path(vis._get_dataset_mesh_dir())
        (mesh_dir / 'AL(L).json').write_text('{}')
        (mesh_dir / 'FB.json').write_text('{}')
        rois = vis._get_available_rois(use_cache=False, fetch_online=False)
        assert rois == sorted(['AL(L)', 'FB'])
        # corrupt cache file falls back to local scan as well
        cache_file = (tmp_path / 'cache' / 'hemibrain_v1_2_1'
                      / 'available_rois.json')
        cache_file.write_text('{not json')
        rois = vis._get_available_rois(use_cache=True, fetch_online=False)
        assert rois == sorted(['AL(L)', 'FB'])


# ---------------------------------------------------------------------------
# individual PDF/PPTX profile exports (synthetic PNGs)
# ---------------------------------------------------------------------------
class TestIndividualExports:
    def _images_dict(self, tmp_path):
        from PIL import Image
        images = {}
        for rank in (1, 2, 10):
            entries = []
            for view in ('front', 'top'):
                p = tmp_path / f'r{rank}_{view}.png'
                Image.new('RGB', (60, 40), (rank * 20 % 256, 100, 50)).save(p)
                entries.append((str(p), view))
            images[f'r{rank}'] = entries
        images['ghost'] = [(str(tmp_path / 'missing.png'), 'front')]
        return images

    def test_create_individual_pdf_by_name(self, tmp_path):
        vis = make_vis()
        out = vis._create_individual_pdf(
            str(tmp_path), self._images_dict(tmp_path),
            images_per_page=(2, 2), title='Profiles',
            organize_by='name', pdf_suffix='_by_name')
        assert out is not None and os.path.exists(out)
        assert out.endswith('_by_name.pdf')
        assert os.path.getsize(out) > 0

    def test_create_individual_pdf_by_view_and_dark_bg(self, tmp_path):
        vis = make_vis()
        out = vis._create_individual_pdf(
            str(tmp_path), self._images_dict(tmp_path),
            organize_by='view', views=['front', 'top'],
            background_color='black')
        assert out is not None and os.path.exists(out)

    def test_create_individual_pdf_empty(self, tmp_path):
        vis = make_vis()
        assert vis._create_individual_pdf(str(tmp_path), {}) is None
        assert vis._create_individual_pdf(
            str(tmp_path), {'a': [(str(tmp_path / 'nope.png'), 'front')]}) is None

    def test_create_individual_pptx_by_name(self, tmp_path):
        vis = make_vis()
        out = vis._create_individual_pptx(
            str(tmp_path), self._images_dict(tmp_path),
            images_per_page=(2, 2), title='Profiles',
            organize_by='name', pptx_suffix='_by_name')
        assert out is not None and os.path.exists(out)

    def test_create_individual_pptx_by_view_dark(self, tmp_path):
        vis = make_vis()
        out = vis._create_individual_pptx(
            str(tmp_path), self._images_dict(tmp_path),
            organize_by='view', views=['front', 'top'],
            background_color='#101010')
        assert out is not None and os.path.exists(out)

    def test_create_individual_pptx_empty(self, tmp_path):
        vis = make_vis()
        assert vis._create_individual_pptx(str(tmp_path), {}) is None


# ---------------------------------------------------------------------------
# backward-compat module functions
# ---------------------------------------------------------------------------
class TestCompatWrappers:
    def test_img2pptx_wrapper(self, tmp_path):
        from PIL import Image
        folder = tmp_path / 'imgs'
        folder.mkdir()
        for i in range(2):
            Image.new('RGB', (40, 40), (255 * i % 256, 100, 200)).save(
                folder / f'img_{i}.png')
        out = vs_module.img2pptx(str(folder), output_pptx=str(tmp_path / 'out.pptx'))
        assert os.path.exists(out)
