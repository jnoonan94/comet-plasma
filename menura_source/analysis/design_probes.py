# import sys
import os
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import random

from menura_utils import *




import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run')
parser.add_argument('--it')
args = parser.parse_args()
#
if args.it != None:
    it = int(args.it)
else:
    it = 0

if args.run == None:
    run_ID = ''
    path = '/home/etienneb/Models/Menura/menura'
else:
    try:
        run_ID = f'{int(args.run):03}'
    except:
        run_ID = args.run

    path = f'/home/etienneb/Models/Menura_own/Data/run_{run_ID}'
# path = '/home/etienneb/Models/Menura_own/menura_ORF'
# path = '/home/etienneb/menura/menura_ORF'
# path = '/home/etienneb/Models/Menura/menura_test_particle'

# path = f'/home/etienneb/Models/Menura_own/Data/run_tmp_2'

# path_remote = '/home/b/behare/Private/menura'
# path_remote = '/home/b/behare/Private/menura_ORF'
# path_remote = '/linkhome/rech/genlag01/ued64ot/menura'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
# if path_remote == '/gpfswork/rech/fuz/ued64ot/run_012':
#     sys.exit('NO, change directory stp!')

remote_label = 'jean-zay'
# remote_label = 'kebnekaise'


class design_probes():

    def __init__(self, md):

        self.md = md

        self.polygon = [[],[]]
        self.line = mpl.lines.Line2D([], [])

        self.fig, self.ax = plt.subplots(figsize=(16, 14))
        self.ax.set_aspect('equal')

        self.ax.add_line(self.line)
        #

        #
        #
        self.fig.canvas.mpl_connect('button_press_event', self.draw_polygon)
        #
        dens = self.md.load_field('dens_tot')
        #
        self.ax.imshow(np.log10(dens).T,
                      extent=[self.md.grid_x_box[0], self.md.grid_x_box[-1],
                              self.md.grid_y_box[0], self.md.grid_y_box[-1]],
                      origin='lower',
                      vmin=-1, vmax=1,
                      cmap=oC.bwr_2,
                      interpolation='bilinear', rasterized=True)

        plt.show()

    # def plot_it(self):



    def draw_polygon(self, event):
        # pass
        if event.inaxes==self.ax:
            self.polygon[0].append(event.xdata)
            self.polygon[1].append(event.ydata)

        else:
            print(np.array(self.polygon).T)
            self.polygon[0].append(self.polygon[0][0])
            self.polygon[1].append(self.polygon[1][0])

        self.line.set_data(self.polygon)

        self.fig.canvas.draw()




md = menura_data(path, path_remote, it, remote_label=remote_label,
                 ask_scp=False, print_param=False, full_scp=True )


if 0:
    ## Designing equilateral tringles.
    probes_spacing = np.array([0.5, 5., 50., 250.])
    probes_positions = np.zeros((2, probes_spacing.size*2+1))

    for p in range(probes_spacing.size):
        probes_positions[0, p*2]   = probes_spacing[p]*np.cos(np.pi*7./6.)
        probes_positions[1, p*2]   = probes_spacing[p]*np.sin(np.pi*7./6.)
        probes_positions[0, p*2+1] = probes_spacing[p]*np.cos(np.pi*5./6.)
        probes_positions[1, p*2+1] = probes_spacing[p]*np.sin(np.pi*5./6.)
    # probes_positions = np.dstack(np.zeros(2), probes_positions)

    probes_positions[0] += 400
    probes_positions[1] += 300

elif 1:
    ## Drawing probes position randomly within a polygon.

    # design_probes(md)
    # sys.exit()

    nb_probes = 20

    probes_positions = []

    from shapely.geometry import Point, Polygon

    # Polygon sheath low
    coords_1 = \
    [(223.19395695, 178.0420996 ),
     (229.99500714, 199.49156558),
     (251.44447312, 214.66313908),
     (271.32446598, 211.00103513),
     (272.8939391 , 197.39893475),
     (260.33815414, 178.0420996 ),
     (239.41184586, 171.76420712)]
    poly_1 = Polygon(coords_1)

    coords_2 = \
    [(23.70297006, 104.16062632),
     (23.70297006, 129.14847099),
     (40.36153317, 140.68132237),
     (59.58295215, 129.14847099),
     (58.30152421, 102.87919839),
     (51.25367059,  77.89135372),
     (29.46939575,  75.96921182)]
    poly_2 = Polygon(coords_2)

    # Polygon sheath high
    coords_3 = \
    [(213.99501791, 279.71625296),
     (233.21643689, 295.7341021 ),
     (255.00071173, 292.53053228),
     (271.01856087, 271.3869714 ),
     (270.37784691, 259.85412002),
     (244.10857431, 256.00983622),
     (221.6835855 , 266.90197364)]
    poly_3 = Polygon(coords_3)

    coords_4 = \
    [(26.26582592, 318.79980487),
     (39.08010524, 347.63193334),
     (63.42723594, 349.55407524),
     (73.03794543, 325.8476585 ),
     (57.66081025, 295.09338814),
     (35.87653541, 294.45267417)]
    poly_4 = Polygon(coords_4)

    # Polygon tail
    coords_5 = \
    [(273.96323639, 237.3858804 ),
     (262.44277551, 244.19342547),
     (245.16208418, 240.00416696),
     (226.83407823, 236.33856577),
     (224.21579167, 230.57833533),
     (231.02333674, 220.10518907),
     (249.35134269, 225.3417622 ),
     (268.72666326, 231.62564996)]
    poly_5 = Polygon(coords_5)

    coords_6 = \
    [(28.82868179, 213.08200051),
     (28.82868179, 231.02199156),
     (45.4872449 , 238.71055915),
     (61.50509404, 225.25556586),
     (61.50509404, 209.87843068),
     (43.565103  , 195.14200947)]
    poly_6 = Polygon(coords_6)

    coords_7 = \
    [(309.05367145, 252.07131114),
     (312.71960685, 254.68983643),
     (315.33813214, 248.9290808 ),
     (312.71960685, 245.2631454 ),
     (309.05367145, 246.83426057)]
    poly_7 = Polygon(coords_7)


    # Polygon upstream (low)
    coords_8 = \
    [(470., 488.),
     (488., 488.),
     (488.,  12.),
     (470.,  12.)]
    poly_8 = Polygon(coords_8)


    coords_1 = np.array(coords_1)
    x_min = np.amin(coords_1[:, 0])
    x_max = np.amax(coords_1[:, 0])
    y_min = np.amin(coords_1[:, 1])
    y_max = np.amax(coords_1[:, 1])
    #
    nb_p=0
    while nb_p<nb_probes:
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        point = Point(x, y)
        if poly_1.contains(point):
            probes_positions.append([x, y])
            nb_p += 1

    coords_2 = np.array(coords_2)
    x_min = np.amin(coords_2[:, 0])
    x_max = np.amax(coords_2[:, 0])
    y_min = np.amin(coords_2[:, 1])
    y_max = np.amax(coords_2[:, 1])
    #
    nb_p=0
    while nb_p<nb_probes:
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        point = Point(x, y)
        if poly_2.contains(point):
            probes_positions.append([x, y])
            nb_p += 1

    coords_3 = np.array(coords_3)
    x_min = np.amin(coords_3[:, 0])
    x_max = np.amax(coords_3[:, 0])
    y_min = np.amin(coords_3[:, 1])
    y_max = np.amax(coords_3[:, 1])
    #
    nb_p=0
    while nb_p<nb_probes:
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        point = Point(x, y)
        if poly_3.contains(point):
            probes_positions.append([x, y])
            nb_p += 1

    coords_4 = np.array(coords_4)
    x_min = np.amin(coords_4[:, 0])
    x_max = np.amax(coords_4[:, 0])
    y_min = np.amin(coords_4[:, 1])
    y_max = np.amax(coords_4[:, 1])
    #
    nb_p=0
    while nb_p<nb_probes:
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        point = Point(x, y)
        if poly_4.contains(point):
            probes_positions.append([x, y])
            nb_p += 1

    coords_5 = np.array(coords_5)
    x_min = np.amin(coords_5[:, 0])
    x_max = np.amax(coords_5[:, 0])
    y_min = np.amin(coords_5[:, 1])
    y_max = np.amax(coords_5[:, 1])
    #
    nb_p=0
    while nb_p<nb_probes:
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        point = Point(x, y)
        if poly_5.contains(point):
            probes_positions.append([x, y])
            nb_p += 1

    coords_6 = np.array(coords_6)
    x_min = np.amin(coords_6[:, 0])
    x_max = np.amax(coords_6[:, 0])
    y_min = np.amin(coords_6[:, 1])
    y_max = np.amax(coords_6[:, 1])
    #
    nb_p=0
    while nb_p<nb_probes:
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        point = Point(x, y)
        if poly_6.contains(point):
            probes_positions.append([x, y])
            nb_p += 1

    coords_7 = np.array(coords_7)
    x_min = np.amin(coords_7[:, 0])
    x_max = np.amax(coords_7[:, 0])
    y_min = np.amin(coords_7[:, 1])
    y_max = np.amax(coords_7[:, 1])
    #
    nb_p=0
    while nb_p<nb_probes:
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        point = Point(x, y)
        if poly_7.contains(point):
            probes_positions.append([x, y])
            nb_p += 1

    coords_8 = np.array(coords_8)
    x_min = np.amin(coords_8[:, 0])
    x_max = np.amax(coords_8[:, 0])
    y_min = np.amin(coords_8[:, 1])
    y_max = np.amax(coords_8[:, 1])
    #
    nb_p=0
    while nb_p<nb_probes:
        x = random.random()*(x_max-x_min)+x_min
        y = random.random()*(y_max-y_min)+y_min
        point = Point(x, y)
        if poly_8.contains(point):
            probes_positions.append([x, y])
            nb_p += 1

    probes_positions = np.array(probes_positions).T
#
np.savetxt('tmp/probes_postions.csv', probes_positions, fmt='%3.2f', delimiter=', ')
#
#
#
# sys.exit()


dens = md.load_field('dens_tot')

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_aspect('equal')

ax.imshow(np.log10(dens).T,
          extent=[md.grid_x_box[0], md.grid_x_box[-1], md.grid_y_box[0], md.grid_y_box[-1]],
          origin='lower',
          vmin=-1, vmax=1,
          cmap=oC.bwr_2,
          interpolation='bilinear', rasterized=True)

for p in range(probes_positions.shape[1]):
    plt.plot(probes_positions[0, p], probes_positions[1, p], 'ko', ms=1)

ax.plot(coords_1[:, 0], coords_1[:, 1], 'k', lw=.5)
ax.plot(coords_2[:, 0], coords_2[:, 1], 'k', lw=.5)
ax.plot(coords_3[:, 0], coords_3[:, 1], 'k', lw=.5)
ax.plot(coords_4[:, 0], coords_4[:, 1], 'k', lw=.5)
ax.plot(coords_5[:, 0], coords_5[:, 1], 'k', lw=.5)
ax.plot(coords_6[:, 0], coords_6[:, 1], 'k', lw=.5)
ax.plot(coords_7[:, 0], coords_7[:, 1], 'k', lw=.5)
ax.plot(coords_8[:, 0], coords_8[:, 1], 'k', lw=.5)

plt.title('log10 density (total)')
oT.set_spines(ax)
plt.tight_layout()
plt.show()
