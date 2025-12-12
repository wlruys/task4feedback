from task4feedback.graphs.mesh.base import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.mesh.plot import *
from task4feedback.graphs.base import *
from task4feedback.graphs.jacobi import *
from task4feedback.interface import TaskTuple
import time

if __name__ == "__main__":
    initialize_gmsh()
    mesh = generate_quad_mesh(L=4, n=3)
    # mesh = generate_tri_mesh(L=4.0, n=5)

    geom = build_geometry(mesh)

    fig, ax = create_mesh_plot(geom, title="Mesh Plot", label_cells=True)

    cells = geom.edge_cell_dict[0]
    map = {}
    side_highlights = {}

    # for edge, cell_neighbors in geom.edge_cell_dict.items():
    #    side_highlights[edge] = {c: random_color(map) for c in cell_neighbors}

    # start_time = time.perf_counter()
    # highlight_artists = highlight_boundary(ax, geom, side_highlights, h=0.2)
    # end_t = time.perf_counter() - start_time
    # print(f"Time taken: {end_t}")

    # jacobi_data = JacobiData.from_mesh(geom)

    jgraph = JacobiGraph(geom, 4)

    for task in jgraph:
        print(task.name)
        print([jgraph.get_name(i) for i in task.dependencies])

    plt.show()

    # highlight_sequence = []
    # for task in jgraph:
    #     input_data = task.read
    #     output_data = task.write
    #     keys = jgraph.data.blocks_to_keys(input_data)
    #     print(f"Task {task.id} reads {keys}")
    #     cell_list, edge_list = filter_key_list(keys)
    #     cell_list = geom_to_int_list(cell_list)

    #     def build_edge_color_dict(edge_list, color="blue"):
    #         edge_dict = {}
    #         for edge, cell in edge_list:
    #             edge = edge.id
    #             cell = cell.id
    #             if edge not in edge_dict:
    #                 edge_dict[edge] = {}
    #             edge_dict[edge][cell] = color
    #         return edge_dict

    #     boundary_highlight = build_edge_color_dict(edge_list)

    #     cell_highlight = {
    #         "red": cell_list,
    #     }

    #     edge_highlight = {}

    #     entry = (cell_highlight, edge_highlight, boundary_highlight)

    #     highlight_sequence.append(entry)

    # print(len(highlight_sequence))
    # print(highlight_sequence[0])

    # ani = animate_highlights(fig, ax, geom, highlight_sequence)

    # G = jgraph.to_networkx()
    # draw(G)

    # plt.tight_layout()
    # plt.show()
    # gmsh.finalize()
    # partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=10)
    # print(partition)
    # shade_partitioning(ax, geom.cell_points, geom.cells, partition)

    # jacobi_data.set_locations_from_list(partition)
    # plt.show()
