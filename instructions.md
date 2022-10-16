`--graph` provides the path of graph(s). It can load a single graph: the path to that graph, or multiple graphs: the path to the folder containing all graphs.

`--model` provides model to use: `Tripling` is ToupleGDD model, `S2V_DQN` is the baseline S2V-DQN model.

`--model_file` provides the path of the model when test, or the file name the model will be saved into when train.

`--test` needs to be specified to test the model.

`--epoch`, `--lr`, `--bs`, `--n_step` are not needed when test.

The node id of graph should be consecutive integers.

`main.py line 54`:

	Set graph type: directed or undirected.
	Set start node id of graph: 0-index or 1-index.

`rl_agents.py line 143`:

	Set the number of epochs to generate initial embedding for graph nodes when train (current 30).

`runner.py line 61`:

	Set the number of epochs to generate initial embedding for graph nodes when test (current 0).

`utils/graph_utils.py line 87-97`:

	Set graph edge weight. The current setting is 1/in-degree(v) for edge (u, v) (in-degree setting).
