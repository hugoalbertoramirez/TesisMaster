from bokeh.plotting import figure
from bokeh.io import output_file, show

p = figure(x_axis_label='fertility (children per woman)',
           y_axis_label='female_literacy (% population)',
           logo=None)

p.circle([1, 2, 3, 4], [4, 3, 2, 1])

#output_file('fert_lit.html')

show(p)

def add(x: int, y: int) -> int:
    return x + y

add('h', 'u')