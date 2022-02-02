from Visualization import ImageVisualizer


def test_results_map():
    info_map = []
    results = []
    for j in range(5):
        for i in range(20):
            info_map.append([None, [j,i], None])
            results.append((j * 20 + i) / (20 * 5))
    vsz = ImageVisualizer(results, info_map, "test_image.png", (225, 225))
    vsz.plot_results_map()
