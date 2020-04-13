import json
import os
import statistics
from collections import Counter

import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch

plotly.io.orca.config.executable = "PATH/TO/orca"

'''
Notes: 73% of videos pass the filter
3.2 seconds and up get annotated
>40 seconds ignored
failure location not in [1,99]% ignored
'''


class MaybeFigure:
    def __init__(self, name, fig_names, figs, gen_fn):
        self.ok = fig_names is None or name in fig_names
        self.name = name
        self.figs = figs
        self.gen_fn = gen_fn

    def __enter__(self):
        return self

    def assign(self, d):
        if self.ok:
            self.figs[self.name] = d

    def gen(self):
        self.assign(self.gen_fn(self.name))

    def __exit__(self, type, value, traceback):
        if self.ok:
            print(f'processed {self.name}')


class FigureBuilder:
    def __init__(self, anns, confusion=None, filetype='pdf', basepath="PATH/TO/tmp",
                 ok_names=None):
        self.anns = anns
        self.filetype = filetype
        self.confusion = confusion
        self.basepath = basepath
        self.ok_names = ok_names
        self.all_names = ['lbl_rel_dist', 'lbl_rel_stdev', 'vid_len', 'kinetics_dist', 'places_dist', 'category_err']
        self.figs = {}
        self.default_layout_kwargs = dict(margin=dict(l=10, r=10, t=55, b=10), height=350,
                                          font=dict(size=26, family='Inter'))
        self.default_layout_x_kwargs = dict(tickfont=dict(size=20))
        self.default_layout_y_kwargs = dict(tickfont=dict(size=20))
        self.default_title_kwargs = dict(x=0.5, xref='container')
        self.maybe_figure = lambda name: MaybeFigure(name, self.ok_names, self.figs, gen_fn=self.gen_fig)

    def __call__(self, *args, **kwargs):
        self.gen_figs()

    def gen_fig(self, name):
        if name == 'confusion':
            M = self.confusion
            M /= M.sum(dim=1).unsqueeze(1)
            M *= 100
            M = M.flip([0]).tolist()
            for i in range(len(M)):
                for j in range(len(M[i])):
                    M[i][j] = round(M[i][j], 2)
            fig = ff.create_annotated_heatmap(z=M, x=['Before failure', 'At failure', 'After failure'],
                                              y=['Before failure', 'At failure', 'After failure'], colorscale='Viridis')
            return {
                'fig': fig,
                'title': 'Classification confusion matrix',
                'xtitle': 'Ground truth',
                'ytitle': '<span>Prediction<br> </span>'
            }
        elif name == 'lbl_rel_dist':
            rel_dist_data = {k: statistics.median(v['rel_t']) for k, v in self.anns.items() if
                0.01 <= statistics.median(v['rel_t']) <= 0.99}
            fig = ff.create_distplot([[_ for _ in rel_dist_data.values()]], [''], show_rug=False, bin_size=0.02,
                                     curve_type='normal', histnorm='probability', show_curve=False)
            fig.update_traces(marker_line_color='rgb(255, 127, 14)', marker_color='rgb(255, 127, 14)',
                              marker_line_width=1,
                              opacity=0.6)
            return {
                'fig': fig,
                'title': '(b) Failure label times',
                'xtitle': 'Time (% of duration)',
                'ytitle': 'Frequency',
                'layout_kwargs': {
                    'showlegend': False
                },
                'layout_x_kwargs': {
                    'tickformat': '%'
                },
                'layout_y_kwargs': {
                    'tickformat': '.1%',
                    'rangemode': 'nonnegative'
                    # 'range': [0, 1]
                },
                'title_kwargs': {
                }
            }
        elif name == 'lbl_rel_stdev':
            rel_stdev_data = {k: statistics.stdev([max(0, _) for _ in v['rel_t']]) for k, v in self.anns.items() if
                0.01 <= statistics.median(v['rel_t']) <= 0.99}
            fig = ff.create_distplot([[_ for _ in rel_stdev_data.values()]], [''], show_rug=False, bin_size=0.01,
                                     histnorm='probability', show_curve=False)
            fig.update_traces(marker_line_color='rgb(44, 160, 44)', marker_color='rgb(44, 160, 44)',
                              marker_line_width=1,
                              opacity=0.6)
            return {
                'fig': fig,
                'title': '(c) Failure label standard deviations',
                'xtitle': 'Deviation (% of duration)',
                'ytitle': 'Frequency',
                'layout_kwargs': {
                    'showlegend': False
                },
                'layout_x_kwargs': {
                    'tickformat': '%'
                },
                'layout_y_kwargs': {
                    'tickformat': '%',
                    'rangemode': 'nonnegative'
                    # 'range': [0, 1]
                },
                'title_kwargs': {
                }
            }
        elif name == 'lbl_stdev':
            stdev_data = {k: statistics.stdev([max(0, _) for _ in v['t']]) for k, v in self.anns.items() if
                0.01 <= statistics.median(v['rel_t']) <= 0.99}
            fig = ff.create_distplot([[(_ if _ < 10 else 10) for _ in stdev_data.values()]], [''], show_rug=False,
                                     bin_size=0.1,
                                     histnorm='probability', show_curve=False)
            fig.update_traces(marker_line_color='rgb(44, 160, 44)', marker_color='rgb(44, 160, 44)',
                              marker_line_width=1,
                              opacity=0.6)
            return {
                'fig': fig,
                'title': 'Failure label standard deviations',
                'xtitle': 'Standard deviation (seconds)',
                'ytitle': 'Frequency',
                'layout_kwargs': {
                    'showlegend': False
                },
                'layout_x_kwargs': {
                    # 'tickformat': '%'
                },
                'layout_y_kwargs': {
                    'tickformat': '.1%',
                    'rangemode': 'nonnegative'
                    # 'range': [0, 1]
                },
                'title_kwargs': {
                }
            }
        elif name == 'vid_len':
            vid_basepath = "PATH/TO/"
            with open(os.path.join(vid_basepath, 'validcliplens.json'), 'r') as fff:
                validcliplens = json.load(fff)

            # ann_lens = {k: v['len'] for k, v in anns.items() if 0.01 <= statistics.median(v['rel_t']) <= 0.99}
            return {
                'fig': ff.create_distplot([[_ for _ in validcliplens if 3.2 <= _ < 30]], [''], show_rug=False,
                                          bin_size=1,
                                          histnorm='probability', show_curve=False),
                'title': '(a) Video lengths',
                'xtitle': 'Length (seconds)',
                'ytitle': 'Frequency',
                'layout_kwargs': {
                    'showlegend': False
                },
                'layout_y_kwargs': {
                    'tickformat': '%',
                    'rangemode': 'nonnegative'
                    # 'range': [0, 1]
                },
                'layout_x_kwargs': {
                    # 'range': [2, 40]
                },
                'title_kwargs': {
                }
            }
        elif name == 'kinetics_dist':
            kinetics_dist = torch.load("PATH/TO/kinetics_dist.pt")
            kinetics_cls = torch.load("PATH/TO/kinetics_classes.pt")
            kinetics_xs, kinetics_ys = zip(
                *Counter({kinetics_cls[k][:25]: v for k, v in kinetics_dist.items()}).most_common())
            fig = go.Figure([go.Bar(x=kinetics_xs, y=kinetics_ys)])
            fig.update_traces(marker_line_color='rgb(214, 39, 40)', marker_color='rgb(214, 39, 40)',
                              marker_line_width=1,
                              opacity=0.6)
            return {
                'fig': fig,
                'title': '(d) Video action classes',
                'xtitle': None,
                'ytitle': 'Frequency',
                'layout_y_kwargs': {
                    'tickformat': '.1%',
                    # 'rangemode': 'nonnegative'
                    # 'range': [0, 1]
                },
                'layout_x_kwargs': {
                    'tickfont': dict(size=11)
                    # 'range': [2, 40]
                },
                'layout_kwargs': {
                    'xaxis_tickangle': 270,
                    'bargap': 0,
                    'height': 370,
                    'font': dict(size=18, family='Inter')
                }
            }
        elif name == 'places_dist':
            with open("PATH/TO/places_dist.json") as f:
                places_dist = json.load(f)
            with open("PATH/TO/categories_places365.txt") as f:
                places_cls = f.readlines()
            places_xs, places_ys = zip(
                *Counter({' - '.join(places_cls[int(k)].split(' ')[0][3:].replace('_', ' ').split('/'))[:25]: v / sum(
                    places_dist.values()) for k, v in places_dist.items()}).most_common()[1:])
            fig = go.Figure([go.Bar(x=places_xs, y=places_ys)])
            fig.update_traces(marker_line_color='rgb(148, 103, 189)', marker_color='rgb(148, 103, 189)',
                              marker_line_width=1,
                              opacity=0.6)
            return {
                'fig': fig,
                'title': '(e) Video scene classes',
                'xtitle': None,
                'ytitle': 'Frequency',
                'layout_y_kwargs': {
                    'tickformat': '.1%',
                    # 'rangemode': 'nonnegative'
                    # 'range': [0, 1]
                },
                'layout_x_kwargs': {
                    'tickfont': dict(size=11)
                    # 'range': [2, 40]
                },
                'layout_kwargs': {
                    'xaxis_tickangle': 270,
                    'bargap': 0,
                    'height': 365,
                    'font': dict(size=18, family='Inter')
                }
            }
        elif name == 'category_err':
            categories = ["Multi-agent", "Single-agent", "Execution error", "Planning error", "Limited visibility",
                "Unexpected", "Environmental", "Limited knowledge", "Limited skill"]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(name='Human', y=categories, x=[3.41, 4.58, 4.81, 14.4, 5.27, 3.26, 2.79, 6.15, 3.48],
                       orientation='h'))
            fig.add_trace(
                go.Bar(name='Kinetics fine-tuning', y=categories, x=[11, 11.9, 7.8, 5.8, 10.6, 14.8, 11.9, 10.5, 9.5],
                       orientation='h'))
            fig.add_trace(
                go.Bar(name='FPS fine-tuning', y=categories, x=[6.8, 13.4, 11.1, 12.4, 16.1, 21.3, 15.4, 13.1, 8.8],
                       orientation='h'))
            # Change the bar mode
            fig.update_layout(barmode='group', legend_orientation="h", legend=dict(y=-0.2), annotations=[
                go.layout.Annotation(
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    text="Error (%)",
                    xref="paper",
                    yref="paper",
                    font=dict(size=25, family='Inter')
                )
            ])
            # fig.show()
            return {
                'fig': fig,
                'title': '',
                'xtitle': '',
                'ytitle': '',
                'layout_y_kwargs': {
                    # 'tickformat': '.1%',
                    # 'rangemode': 'nonnegative'
                    # 'range': [0, 1]
                    'tickfont': dict(size=26)
                },
                'layout_x_kwargs': {
                    # 'tickfont': dict(size=11)
                    # 'range': [2, 40]
                },
                'layout_kwargs': {
                    # 'xaxis_tickangle': 90,
                    # 'bargap': 0,
                    'height': 650,
                    'margin': dict(l=0, r=10, t=10, b=200),
                    'font': dict(size=25, family='Inter')
                }
            }

    def gen_figs(self):
        for name in self.all_names:
            with self.maybe_figure(name) as f:
                f.gen()

        for k, v in self.figs.items():
            fig = v['fig']
            fig.update_layout(
                title=go.layout.Title(
                    text=v['title'],
                    **{**self.default_title_kwargs,
                        **v.get('title_kwargs', {})},
                ) if 'title' in v else None,
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(
                        text=v['xtitle']
                    ),
                    **{**self.default_layout_x_kwargs,
                        **v.get('layout_x_kwargs', {})},
                ),
                yaxis=go.layout.YAxis(
                    title=go.layout.yaxis.Title(
                        text=v['ytitle']
                    ),
                    **{**self.default_layout_y_kwargs,
                        **v.get('layout_y_kwargs', {})},
                ),
                **{**self.default_layout_kwargs,
                    **v.get('layout_kwargs', {})},
            )
            fig.update_yaxes(automargin=True)
            fig_fn = os.path.join(self.basepath, f'{k}.{self.filetype}')
            print(f'writing {fig_fn}')
            fig.write_image(fig_fn)
            print('done!')
