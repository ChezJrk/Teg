from teg.eval import evaluate
import numpy as np
import png

from test_utils import progress_bar


def render_image(expr, variables=(), res=(64, 64), bounds=((0, 1), (0, 1)), bindings={}, silent=False):
    assert len(variables) == 2, 'Exactly two variable-pairs required'

    image = np.zeros(res)
    _xs = np.linspace(bounds[0][0], bounds[0][1], res[0] + 1)
    _ys = np.linspace(bounds[1][0], bounds[1][1], res[1] + 1)

    for nx, (x_lb, x_ub) in enumerate(zip(_xs, _xs[1:])):
        for ny, (y_lb, y_ub) in enumerate(zip(_ys, _ys[1:])):
            if (nx == 0) and (ny == 0) and not silent:
                print('Compiling to C...')
            value = evaluate(expr, bindings={**bindings,
                                             variables[0][0]: x_lb,
                                             variables[0][1]: x_ub,
                                             variables[1][0]: y_lb,
                                             variables[1][1]: y_ub},
                             num_samples=20, backend='C_PyBind')
            image[nx, ny] = value
            if not silent:
                progress_bar(nx * res[1] + ny + 1, res[0] * res[1], prefix=f'Rendering {res[0]}x{res[1]} plot')

    return image


def save_image(image, filename):
    image = image.T[::-1, :].copy()
    image = ((image / np.max(image)) * 255.0).astype(np.uint8)
    png.from_array(image, mode="L").save(filename)
