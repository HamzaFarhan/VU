import argparse
import json
import math

import dash
import networkx as nx
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

from vu_models import Topics

TOPICS_FILE = "math_topics_4.json"
EDGE_COLOR = "purple"


def find_points_on_line(
    p1: tuple[float, float], p2: tuple[float, float], node_radius_scaled: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    direction_vector = np.array(p2) - np.array(p1)
    if np.linalg.norm(direction_vector) <= 0:
        print("np.linalg.norm(direction_vector): ", np.linalg.norm(direction_vector))
        print("p1, p2; line 9: ", p1, p2)
    normalized_direction = direction_vector / np.linalg.norm(direction_vector)
    delta = 1.2 * 0.43 * node_radius_scaled
    p3 = tuple(np.array(p1) + delta * normalized_direction)
    p4 = tuple(np.array(p2) - delta * normalized_direction)
    return p3, p4


def calculate_subtopic_positions(
    topics: Topics, topic_positions, circle_radius: float = 0.3
) -> dict[str, tuple[float, float]]:
    subtopic_positions = {}
    for topic_id, topic in topics.topics.items():
        topic_position = topic_positions[topic_id]
        num_subtopics = len(topic.subtopics)
        for i, subtopic_id in enumerate(topic.subtopics):
            angle = 2 * math.pi * ((i + 15) / num_subtopics)
            x = topic_position[0] + 0.6 * circle_radius * math.cos(angle)
            y = topic_position[1] + 0.6 * circle_radius * math.sin(angle)
            subtopic_positions[subtopic_id] = (x, y)
    return subtopic_positions


def create_arrowhead_list(
    graph: nx.Graph, pos: dict, selected_node: str, node_radius_scaled: float
) -> list[dict]:
    arrowhead_list = []
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        p1 = (x0, y0)
        p2 = (x1, y1)
        e1, e2 = find_points_on_line(p1, p2, node_radius_scaled)
        x_arrowhead = e2[0]
        y_arrowhead = e2[1]

        x_arrowhead_start = e1[0]
        y_arrowhead_start = e1[1]
        # Append edge coordinates to the edge_trace
        edge_x += [x_arrowhead_start, x_arrowhead, None]
        edge_y += [y_arrowhead_start, y_arrowhead, None]
        arrowhead = dict(
            x=x_arrowhead,
            y=y_arrowhead,
            xref="x",
            yref="y",
            axref="x",  # Set the reference for the arrowhead starting point to 'x'
            ayref="y",
            ax=x_arrowhead_start,
            ay=y_arrowhead_start,
            showarrow=True,
            arrowhead=2,  # Use arrowhead style 2 (a filled arrowhead)
            arrowsize=1.0,
            arrowwidth=1.0,
            arrowcolor=EDGE_COLOR,
        )
        arrowhead_list.append(arrowhead)  # Add the arrowhead to the trace

    if selected_node:
        # print ("In create_arrowhead_list; selected_node; line 40: ", selected_node)

        arrowhead_list = []
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            p1 = (x0, y0)
            p2 = (x1, y1)
            e1, e2 = find_points_on_line(p1, p2, node_radius_scaled)
            x_arrowhead = e2[0]
            y_arrowhead = e2[1]

            x_arrowhead_start = e1[0]
            y_arrowhead_start = e1[1]
            # Append edge coordinates to the edge_trace
            edge_x += [x_arrowhead_start, x_arrowhead, None]
            edge_y += [y_arrowhead_start, y_arrowhead, None]
            arrowhead = dict(
                x=x_arrowhead,
                y=y_arrowhead,
                xref="x",
                yref="y",
                axref="x",  # Set the reference for the arrowhead starting point to 'x'
                ayref="y",
                ax=x_arrowhead_start,
                ay=y_arrowhead_start,
                showarrow=True,
                arrowhead=2,  # Use arrowhead style 2 (a filled arrowhead)
                arrowsize=1.0,
                arrowwidth=1.0,
                arrowcolor=EDGE_COLOR,
            )
            arrowhead_list.append(arrowhead)  # Add the arrowhead to the trace

    return arrowhead_list


def create_topics_subtopics_network_graph(
    topics: Topics, selected: str = ""
) -> go.Figure:
    topic_node_size = 180
    subtopic_node_size = 40
    topics_graph = nx.DiGraph()
    subtopics_graph = nx.DiGraph()

    node_sizes = {}
    topic_texts = []
    subtopic_texts = []
    for topic_id, topic in topics.topics.items():
        topic_texts.append(topic_id)
        topics_graph.add_node(topic_id, type="topic", text=topic_id, label=topic.topic)
        node_sizes[topic_id] = topic_node_size
        for subtopic_id, subtopic in topic.subtopics.items():
            subtopic_texts.append(subtopic_id)
            subtopics_graph.add_node(
                subtopic_id,
                type="subtopic",
                text=subtopic_id,
                label=subtopic.subtopic,
                color="red",
            )
            node_sizes[subtopic_id] = subtopic_node_size
    for topic_id, topic in topics.topics.items():
        for prereq_id in topic.prerequisite_ids:
            if prereq_id.count("_") > 0:
                continue
            topics_graph.add_edge(
                prereq_id, topic_id, type="prereq", color=EDGE_COLOR, arrow=True
            )
    for subtopic_id, subtopic in topics.subtopics.items():
        for prereq_id in subtopic.prerequisite_ids:
            if prereq_id.count("_") > 1:
                continue
            subtopics_graph.add_edge(
                prereq_id, subtopic_id, type="prereq", color=EDGE_COLOR, arrow=True
            )
    topic_positions = nx.circular_layout(topics_graph)
    subtopic_positions = calculate_subtopic_positions(
        topics=topics, topic_positions=topic_positions
    )
    all_positions = {**topic_positions, **subtopic_positions}
    height = 1000
    width = 1000
    x_range = (
        max(all_positions.values(), key=lambda x: x[0])[0]
        - min(all_positions.values(), key=lambda x: x[0])[0]
    )
    y_range = (
        max(all_positions.values(), key=lambda x: x[1])[1]
        - min(all_positions.values(), key=lambda x: x[1])[1]
    )
    x_scale = width / x_range
    y_scale = height / y_range
    node_topics_radius = 0.5 * topic_node_size
    node_radius_topics_scaled = 0.5 * (x_range + y_range) * node_topics_radius
    node_subtopics_radius = 0.5 * subtopic_node_size
    node_radius_subtopics_scaled = 0.5 * (x_range + y_range) * node_subtopics_radius
    scaled_positions = {
        node: (
            (x - min(all_positions.values(), key=lambda x: x[0])[0]) * x_scale,
            (y - min(all_positions.values(), key=lambda x: x[1])[1]) * y_scale,
        )
        for node, (x, y) in all_positions.items()
    }
    pos = scaled_positions
    node_x_topics = [pos[node][0] for node in topics_graph.nodes()]
    node_y_topics = [pos[node][1] for node in topics_graph.nodes()]
    # print(f"subtopics_graph.nodes(): {subtopics_graph.nodes()}")
    node_x_subtopics = [pos[node][0] for node in subtopics_graph.nodes()]
    node_y_subtopics = [pos[node][1] for node in subtopics_graph.nodes()]
    node_trace_topics = go.Scatter(
        x=node_x_topics,
        y=node_y_topics,
        text=topic_texts,
        mode="markers+text",
        hoverinfo="text",
        customdata=["topic"],
        marker=dict(
            size=[node_sizes[node] for node in topics_graph.nodes()],
            color="black",
            opacity=0.5,
            line=dict(width=1, color="black"),
            symbol="circle-open",
        ),
    )
    node_trace_subtopics = go.Scatter(
        x=node_x_subtopics,
        y=node_y_subtopics,
        text=subtopic_texts,
        mode="markers",
        hoverinfo="text",
        customdata=["subtopic"],
        marker=dict(
            size=[node_sizes[node] for node in subtopics_graph.nodes()],
            color="rgba(255,0,0,1)",
            opacity=1,
            line=dict(width=1, color="red"),
            symbol="circle-open",
        ),
    )
    arrowhead_trace = go.Scatter(
        x=[], y=[], mode="markers", hoverinfo="none", showlegend=False
    )
    if selected == "All Topics":
        arrowhead_topics_list = create_arrowhead_list(
            topics_graph, pos, selected, node_radius_topics_scaled
        )
        fig = go.Figure(
            data=[arrowhead_trace, node_trace_topics, node_trace_subtopics],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    fixedrange=True,
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    fixedrange=True,
                ),
                height=height,
                width=width,
            ),
        )

        fig.update_layout(annotations=arrowhead_topics_list)
    if selected == "All Subtopics":
        arrowhead_subtopics_list = create_arrowhead_list(
            subtopics_graph, pos, selected, node_radius_subtopics_scaled
        )
        fig = go.Figure(
            data=[arrowhead_trace, node_trace_topics, node_trace_subtopics],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    fixedrange=True,
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    fixedrange=True,
                ),
                height=900,
                width=900,
            ),
        )
        fig.update_layout(annotations=arrowhead_subtopics_list)
    fig.update_traces(textposition="top center")
    fig.update_layout(
        title="Topics", title_x=0.5, hovermode="closest", showlegend=False
    )
    return fig


# topics = Topics(**json.load(open(TOPICS_FILE)))
# # print(f"subtopics: {topics.subtopics.keys()}")
# app = dash.Dash(__name__)
# selected_topic = "All Topics"
# options = [
#     {"label": "All Topics", "value": "All Topics"},
#     {"label": "All Subtopics", "value": "All Subtopics"},
# ]
# app.layout = html.Div(
#     [
#         html.H1(""),
#         dcc.Dropdown(
#             id="topic-dropdown",
#             options=options,
#             value="All Topics",
#         ),
#         dcc.Graph(
#             id="topic-dependencies-graph",
#             figure=create_topics_subtopics_network_graph(
#                 topics=topics, selected="All Subtopics"
#             ),
#         ),
#     ],
#     style={"textAlign": "top", "width": "60%", "margin": "auto"},
# )


def update_dropdown_value(click_data, dropdown_options):
    print("In update_dropdown_value; Inputs are:")
    if click_data:
        print("In update_dropdown_value; click_data: ", click_data)
    print("dropdown_options: ", dropdown_options)
    if (
        click_data is not None
        and "points" in click_data
        and len(click_data["points"]) > 0
    ):
        # Check if the click is on a small blue circle (subtopic)
        if click_data["points"][0]["curveNumber"] == 6:
            # Safely access the 'customdata' value using the .get() method
            customdata = click_data["points"][0].get("customdata")

            if customdata is not None:
                # Check if customdata matches any of the dropdown values
                selected_value = next(
                    (
                        option["value"]
                        for option in dropdown_options
                        if option["value"] == customdata
                    ),
                    None,
                )
                if selected_value is not None:
                    return selected_value

        # For large black circles (main topics), use the 'text' attribute instead of 'customdata'
        clicked_label = click_data["points"][0].get("text")
        if clicked_label is not None:
            # Check if clicked_label matches any of the dropdown values
            selected_value = next(
                (
                    option["value"]
                    for option in dropdown_options
                    if option["label"] == clicked_label
                ),
                None,
            )
            if selected_value is not None:
                return selected_value

    # If no click data or no corresponding value found, do not update the dropdown value
    return dash.no_update


def create_app(topics_file: str):
    topics = Topics(**json.load(open(topics_file)))
    app = dash.Dash(__name__)

    options = [
        {"label": "All Topics", "value": "All Topics"},
        {"label": "All Subtopics", "value": "All Subtopics"},
    ]

    app.layout = html.Div(
        [
            html.H1(""),
            dcc.Dropdown(
                id="topic-dropdown",
                options=options,
                value="All Topics",
            ),
            dcc.Graph(
                id="topic-dependencies-graph",
                figure=create_topics_subtopics_network_graph(
                    topics=topics, selected="All Subtopics"
                ),
            ),
        ],
        style={"textAlign": "top", "width": "60%", "margin": "auto"},
    )

    @app.callback(
        Output("topic-dependencies-graph", "figure"),
        [Input("topic-dropdown", "value")],
    )
    def update_graph(selected):
        fig = create_topics_subtopics_network_graph(topics=topics, selected=selected)
        return fig

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Run the dashboard with specified topics file and port."
    )
    parser.add_argument("-f", "--topics_file", help="Path to the topics JSON file")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8050,
        help="Port to run the Dash app on (default: 8050)",
    )

    args = parser.parse_args()

    app = create_app(args.topics_file)
    app.run_server(port=args.port)


if __name__ == "__main__":
    main()
