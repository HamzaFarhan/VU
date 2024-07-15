import json

import dash
import networkx as nx
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

from vu_models import Topics


def create_graph_from_topics(topics: Topics):
    graphs = []
    for topic in topics.topics.values():
        G = nx.DiGraph()
        G.add_node(topic.id, type="topic", label=topic.topic)
        for subtopic in topic.subtopics.values():
            G.add_node(subtopic.id, type="subtopic")  # , label=subtopic.subtopic)
            G.add_edge(topic.id, subtopic.id)
            for concept in subtopic.concepts.values():
                G.add_node(
                    concept.id, type="learning outcome"
                )  # , label=concept.concept)
                G.add_edge(subtopic.id, concept.id)
                for question in concept.questions.values():
                    G.add_node(
                        question.id,
                        type="question",
                        # label=question.question_number,
                    )
                    G.add_edge(concept.id, question.id)
        # for topic in topics.topics.values():
        for prereq in topic.prerequisite_ids:
            G.add_edge(
                prereq,
                topic.id,
                type="prerequisite",
                color="red",
                arrow=True,
            )
        for subtopic in topic.subtopics.values():
            for prereq in subtopic.prerequisite_ids:
                G.add_edge(
                    prereq,
                    subtopic.id,
                    type="prerequisite",
                    color="red",
                    arrow=True,
                )
            for concept in subtopic.concepts.values():
                for prereq in concept.prerequisite_ids:
                    G.add_edge(
                        prereq,
                        concept.id,
                        type="prerequisite",
                        color="red",
                        arrow=True,
                    )
                for question in concept.questions.values():
                    for prereq in question.prerequisite_ids:
                        G.add_edge(
                            prereq,
                            question.id,
                            type="prerequisite",
                            color="red",
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
        line=dict(width=1, color="red"),
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
                prereq_edge_trace["x"] += (x0, x1, None)
                prereq_edge_trace["y"] += (y0, y1, None)
        else:
            edge_trace["x"] += (x0, x1, None)
            edge_trace["y"] += (y0, y1, None)

    node_traces = []
    colors = {
        "topic": "red",
        "subtopic": "blue",
        "learning outcome": "green",
        "question": "orange",
    }
    sizes = {"topic": 15, "subtopic": 12, "learning outcome": 10, "question": 8}

    node_indices = {}
    index = 0

    for node_type in ["topic", "subtopic", "learning outcome", "question"]:
        node_trace = go.Scatter(
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

        for node in G.nodes():
            if G.nodes[node].get("type") == node_type:
                x, y = pos[node]
                node_trace["x"] += (x,)
                node_trace["y"] += (y,)
                # node_info = (
                #     f"{G.nodes[node]['type'].capitalize()}: {G.nodes[node]['label']}"
                # )
                node_trace["text"] += (node,)  # (node_info,)
                node_indices[node] = node
                index += 1

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


dummy_topics = Topics(**json.load(open("math_dummy.json")))

initial_fig, G, pos, node_indices = create_graph_visualization(dummy_topics)

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
    print("UPDATE GRAPH CALLED")
    pos = graph_data["pos"]
    node_indices = graph_data["node_indices"]
    selected_node = graph_data["selected_node"]

    if clickData:
        # print(f'CLICK DATA = {clickData}, selected_node = {selected_node}')
        point = clickData["points"][0]
        point_index = str(point["text"])
        try:
            clicked_node = node_indices[point_index]
            # print(f'CLICKED NODE = {clicked_node}, SELECTED NODE = {selected_node}')
            if clicked_node == selected_node:
                selected_node = None
            else:
                selected_node = clicked_node
            # print(f'FINAL SELECTED NODE = {selected_node}')
        except KeyError:
            print(f"KeyError: Unable to find node for point_index: {point_index}")
            print("Available indices:", list(node_indices.keys()))
    # print('UPDATING FIGURE')
    updated_fig, _, new_pos, new_node_indices = create_graph_visualization(
        dummy_topics, selected_node
    )
    # print(f'New Pos = {new_pos}, New Node Indices = {new_node_indices}, selected node = {selected_node}')
    return updated_fig, {
        "pos": new_pos,
        "node_indices": new_node_indices,
        "selected_node": selected_node,
    }


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
