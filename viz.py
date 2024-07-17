import argparse
import json

import dash
import networkx as nx
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

from vu_models import Topics

PREREQ_COLOR = "purple"


def create_graph_from_topics(topics: Topics):
    graphs = []
    for topic_id, topic in topics.topics.items():
        topic_id = topics.id_to_numbered(topic_id)
        G = nx.DiGraph()
        G.add_node(topic_id, type="topic", label=topic.topic)
        for subtopic_id, subtopic in topic.subtopics.items():
            subtopic_id = topics.id_to_numbered(subtopic_id)
            G.add_node(subtopic_id, type="subtopic", label=subtopic.subtopic)
            G.add_edge(topic_id, subtopic_id)
            for concept_id, concept in subtopic.concepts.items():
                concept_id = topics.id_to_numbered(concept_id)
                G.add_node(concept_id, type="learning outcome", label=concept.concept)
                G.add_edge(subtopic_id, concept_id)
                for question in concept.questions.values():
                    question_id = f"{concept_id}.{question.question_number}"
                    G.add_node(question_id, type="question", label=question.problem)
                    G.add_edge(concept_id, question_id)
        for prereq in topic.prerequisite_ids:
            if prereq.count("_") > 0:
                continue
            G.add_edge(
                topics.id_to_numbered(prereq),
                topic_id,
                type="prerequisite",
                color=PREREQ_COLOR,
                arrow=True,
            )
        for subtopic_id, subtopic in topic.subtopics.items():
            for prereq in subtopic.prerequisite_ids:
                if prereq.count("_") > 1:
                    continue
                G.add_edge(
                    topics.id_to_numbered(prereq),
                    topics.id_to_numbered(subtopic_id),
                    type="prerequisite",
                    color=PREREQ_COLOR,
                    arrow=True,
                )
            for concept_id, concept in subtopic.concepts.items():
                concept_id = topics.id_to_numbered(concept_id)
                for prereq in concept.prerequisite_ids:
                    if prereq.count("_") > 2:
                        continue
                    G.add_edge(
                        topics.id_to_numbered(prereq),
                        concept_id,
                        type="prerequisite",
                        color=PREREQ_COLOR,
                        arrow=True,
                    )
                for question in concept.questions.values():
                    question_id = f"{concept_id}.{question.question_number}"
                    for prereq in question.prerequisite_ids:
                        if prereq.count("_") > 3:
                            continue
                        G.add_edge(
                            topics.id_to_numbered(prereq),
                            question_id,
                            type="prerequisite",
                            color=PREREQ_COLOR,
                            arrow=True,
                        )
        graphs.append(G)
    return nx.compose_all(graphs)


def create_graph_visualization(topics: Topics, selected_node=None):
    G = create_graph_from_topics(topics)

    pos = nx.nx_pydot.graphviz_layout(G, prog="neato")

    x_values = [x for x, y in pos.values()]
    y_values = [y for x, y in pos.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    for node in pos:
        x, y = pos[node]
        pos[node] = ((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min))

    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
    )

    prereq_edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color=PREREQ_COLOR),
        hoverinfo="none",
        mode="lines+markers",
        marker=dict(symbol="arrow", size=8, angleref="previous"),
    )

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        if edge[2].get("type") == "prerequisite":
            # print(f'PREREQUISITE EDGE FOUND: SELECTED NODE = {selected_node}, EDGE = {edge}')
            if selected_node and (edge[0] == selected_node or edge[1] == selected_node):
                # print(f'DRAWING SELECTED NODE EDGE: SELECTED NODE = {selected_node}')
                prereq_edge_trace["x"] += (x0, x1, None)  # type: ignore
                prereq_edge_trace["y"] += (y0, y1, None)  # type: ignore
        else:
            edge_trace["x"] += (x0, x1, None)  # type: ignore
            edge_trace["y"] += (y0, y1, None)  # type: ignore

    node_traces = []
    colors = {
        "topic": "red",
        "subtopic": "blue",
        "learning outcome": "green",
        "question": "orange",
    }
    sizes = {"topic": 15, "subtopic": 12, "learning outcome": 10, "question": 8}

    node_indices = {}

    for node_type in ["topic", "subtopic", "learning outcome", "question"]:
        node_trace_args = dict(
            x=[],
            y=[],
            text=[],
            mode="markers",
            textposition="top center",
            hoverinfo="text",
            name=node_type.capitalize(),
            marker=dict(
                size=sizes[node_type], color=colors[node_type], symbol="circle"
            ),
            textfont=dict(size=sizes[node_type] - 2),
        )
        node_trace = go.Scatter(**node_trace_args)
        for node in G.nodes():
            g_node_type = G.nodes[node].get("type")
            if g_node_type == node_type:
                x, y = pos[node]
                node_trace["x"] += (x,)  # type: ignore
                node_trace["y"] += (y,)  # type: ignore

                if node_type == "question":
                    # Include topic, subtopic, and learning outcome for questions
                    topic_id = ".".join(node.split(".")[:1])
                    subtopic_id = ".".join(node.split(".")[:2])
                    learning_outcome_id = ".".join(node.split(".")[:3])
                    topic_name = G.nodes[topic_id]["label"]
                    subtopic_name = G.nodes[subtopic_id]["label"]
                    learning_outcome_name = G.nodes[learning_outcome_id]["label"]
                    question_label = G.nodes[node]["label"]
                    node_info = f"Question: {question_label}\n(Topic: {topic_name}, Subtopic: {subtopic_name}, Learning Outcome: {learning_outcome_name})"
                elif node_type == "subtopic":
                    # Include topic name for subtopics
                    topic_id = ".".join(node.split(".")[:1])
                    topic_name = G.nodes[topic_id]["label"]
                    node_info = (
                        f"Subtopic: {G.nodes[node]['label']} (Topic: {topic_name})"
                    )
                elif node_type == "learning outcome":
                    # Include topic and subtopic names for learning outcomes
                    topic_id = ".".join(node.split(".")[:1])
                    subtopic_id = ".".join(node.split(".")[:2])
                    topic_name = G.nodes[topic_id]["label"]
                    subtopic_name = G.nodes[subtopic_id]["label"]
                    node_info = f"Learning Outcome: {G.nodes[node]['label']} (Topic: {topic_name}, Subtopic: {subtopic_name})"
                else:
                    node_info = f"{G.nodes[node]['type'].capitalize()}: {G.nodes[node]['label']}"

                node_trace["text"] += (node_info,)  # type: ignore
                node_indices[node_info] = node

        node_traces.append(node_trace)

    fig = go.Figure(
        data=[edge_trace, prereq_edge_trace] + node_traces,
        layout=go.Layout(
            title="Topic Structure",
            titlefont_size=16,
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            width=1200,
            legend=dict(x=1.05, y=1, traceorder="normal"),
            dragmode="select",
        ),
    )

    return fig, G, pos, node_indices


def create_app(topics_file: str):
    topics = Topics(**json.load(open(topics_file)))
    initial_fig, G, pos, node_indices = create_graph_visualization(topics)

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id="topic-graph", figure=initial_fig),
            dcc.Store(
                id="graph-data",
                data={
                    "pos": pos,
                    "node_indices": node_indices,
                    "selected_node": None,
                },
            ),
        ]
    )

    @app.callback(
        Output("topic-graph", "figure"),
        Output("graph-data", "data"),
        Input("topic-graph", "clickData"),
        State("graph-data", "data"),
    )
    def update_graph(clickData, graph_data):
        node_indices = graph_data["node_indices"]
        selected_node = graph_data["selected_node"]

        if clickData:
            point = clickData["points"][0]
            point_index = str(point["text"])
            try:
                clicked_node = node_indices[point_index]
                if clicked_node == selected_node:
                    selected_node = None
                else:
                    selected_node = clicked_node
            except KeyError:
                print(f"KeyError: Unable to find node for point_index: {point_index}")
                print("Available indices:", list(node_indices.keys()))

        updated_fig, _, new_pos, new_node_indices = create_graph_visualization(
            topics, selected_node
        )
        return updated_fig, {
            "pos": new_pos,
            "node_indices": new_node_indices,
            "selected_node": selected_node,
        }

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Run the topic visualization dashboard."
    )
    parser.add_argument(
        "-f",
        "--topics_file",
        default="math_topics_4.json",
        help="Path to the topics JSON file",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8051,
        help="Port to run the Dash app on (default: 8051)",
    )

    args = parser.parse_args()

    app = create_app(args.topics_file)
    app.run_server(debug=True, port=args.port)


if __name__ == "__main__":
    main()
