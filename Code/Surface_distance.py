import argparse
import logging
import os
import sys

import nibabel as nib
import numpy as np
import scipy.ndimage
from sklearn import metrics

# neighbour_code_to_normals is a lookup table.
# For every binary neighbour code
# (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes)
# it contains the surface normals of the triangles (called "surfel" for
# "surface element" in the following). The length of the normal
# vector encodes the surfel area.
#
# created by compute_surface_area_lookup_table.ipynb using the
# marching_cube algorithm, see e.g. https://en.wikipedia.org/wiki/Marching_cubes
#
neighbour_code_to_normals = [
	[[0, 0, 0]],
	[[0.125, 0.125, 0.125]],
	[[-0.125, -0.125, 0.125]],
	[[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
	[[0.125, -0.125, 0.125]],
	[[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
	[[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
	[[-0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
	[[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
	[[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
	[[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
	[[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
	[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
	[[0.125, -0.125, -0.125]],
	[[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
	[[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
	[[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
	[[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25],
	 [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
	[[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
	[[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
	[[0.125, 0.125, 0.125], [0.375, 0.375, 0.375],
	 [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
	[[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
	[[0.375, 0.375, 0.375], [0.0, 0.25, -0.25],
	 [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
	[[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125],
	 [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
	[[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
	[[0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
	[[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
	[[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
	[[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
	[[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25],
	 [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
	[[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
	[[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[0.25, 0.25, -0.25], [0.25, 0.25, -0.25],
	 [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
	[[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
	[[0.5, 0.0, 0.0], [0.25, -0.25, 0.25],
	 [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
	[[0.0, 0.25, -0.25], [0.375, -0.375, -0.375],
	 [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
	[[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
	[[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
	[[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
	[[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
	[[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
	[[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25],
	 [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
	[[0.125, 0.125, 0.125], [0.0, -0.5, 0.0],
	 [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
	[[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
	[[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
	[[0.0, 0.5, 0.0], [0.25, 0.25, -0.25],
	 [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0],
	 [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
	[[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
	[[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0],
	 [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
	[[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
	[[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
	[[-0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
	[[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
	[[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
	[[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[0.375, -0.375, 0.375], [0.0, -0.25, -0.25],
	 [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
	[[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
	[[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
	 [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
	[[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
	 [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
	[[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375],
	 [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
	[[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
	[[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
	[[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
	[[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375],
	 [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
	[[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
	[[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
	[[0.125, 0.125, 0.125], [0.125, 0.125, 0.125],
	 [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
	[[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
	[[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
	[[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
	 [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25],
	 [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
	[[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
	[[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375],
	 [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
	[[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
	[[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
	[[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
	[[0.0, -0.5, 0.0], [0.125, 0.125, -0.125],
	 [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
	[[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
	[[0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
	 [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.0, -0.25, 0.25], [0.0, -0.25, 0.25],
	 [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
	[[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
	[[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
	 [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
	[[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25],
	 [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25],
	 [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
	[[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125],
	 [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
	[[0.375, -0.375, 0.375], [0.0, 0.25, 0.25],
	 [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
	[[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
	[[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0],
	 [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
	[[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
	[[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
	[[0.5, 0.0, -0.0], [0.25, 0.25, 0.25],
	 [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
	[[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
	[[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
	[[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
	[[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
	[[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
	[[0.5, 0.0, -0.0], [0.25, 0.25, 0.25],
	 [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
	[[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
	[[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
	[[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
	[[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0],
	 [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
	[[0.375, -0.375, 0.375], [0.0, 0.25, 0.25],
	 [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
	[[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125],
	 [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
	[[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25],
	 [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25],
	 [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
	 [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
	[[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
	[[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
	[[0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
	 [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
	[[0.0, -0.5, 0.0], [0.125, 0.125, -0.125],
	 [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
	[[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
	[[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
	[[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375],
	 [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
	[[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
	[[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25],
	 [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
	[[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
	 [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
	[[0.125, 0.125, 0.125], [0.125, 0.125, 0.125],
	 [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
	[[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
	[[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
	[[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375],
	 [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
	[[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
	[[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
	[[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
	[[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375],
	 [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
	[[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
	 [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
	[[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
	[[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
	 [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
	[[0.375, -0.375, 0.375], [0.0, -0.25, -0.25],
	 [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
	[[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
	[[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
	[[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
	[[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.125, -0.125, 0.125]],
	[[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
	[[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
	[[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
	[[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0],
	 [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
	[[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
	[[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0],
	 [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
	[[0.0, 0.5, 0.0], [0.25, 0.25, -0.25],
	 [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
	[[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
	[[0.125, 0.125, 0.125], [0.0, -0.5, 0.0],
	 [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
	[[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25],
	 [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
	[[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
	[[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
	[[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
	[[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
	[[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
	[[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[0.0, 0.25, -0.25], [0.375, -0.375, -0.375],
	 [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
	[[0.5, 0.0, 0.0], [0.25, -0.25, 0.25],
	 [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
	[[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
	[[0.25, 0.25, -0.25], [0.25, 0.25, -0.25],
	 [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
	[[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
	[[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
	[[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25],
	 [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
	[[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
	[[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
	[[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
	[[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
	[[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
	[[0.125, -0.125, 0.125]],
	[[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
	[[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125],
	 [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
	[[0.375, 0.375, 0.375], [0.0, 0.25, -0.25],
	 [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
	[[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
	[[0.125, 0.125, 0.125], [0.375, 0.375, 0.375],
	 [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
	[[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
	[[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
	[[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
	[[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25],
	 [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
	[[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
	[[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
	[[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
	[[0.125, -0.125, -0.125]],
	[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
	[[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
	[[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
	[[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
	[[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
	[[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
	[[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
	[[-0.125, 0.125, 0.125]],
	[[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
	[[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
	[[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
	[[0.125, -0.125, 0.125]],
	[[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
	[[-0.125, -0.125, 0.125]],
	[[0.125, 0.125, 0.125]],
	[[0, 0, 0]]]


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
	"""Compute closest distances from all surface points to the other surface.

	Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
	the predicted mask `mask_pred`, computes their area in mm^2 and the distance
	to the closest point on the other surface. It returns two sorted lists of
	distances together with the corresponding surfel areas. If one of the masks
	is empty, the corresponding lists are empty and all distances in the other
	list are `inf`

	Args:
	  mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
	  mask_pred: 3-dim Numpy array of type bool. The predicted mask.
	  spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
		  direction

	Returns:
	  A dict with
	  "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
		  from all ground truth surface elements to the predicted surface,
		  sorted from smallest to largest
	  "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
		  from all predicted surface elements to the ground truth surface,
		  sorted from smallest to largest
	  "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of
		  the ground truth surface elements in the same order as
		  distances_gt_to_pred
	  "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of
		  the predicted surface elements in the same order as
		  distances_pred_to_gt

	"""

	# compute the area for all 256 possible surface elements
	# (given a 2x2x2 neighbourhood) according to the spacing_mm
	neighbour_code_to_surface_area = np.zeros([256])
	for code in range(256):
		normals = np.array(neighbour_code_to_normals[code])
		sum_area = 0
		for normal_idx in range(normals.shape[0]):
			# normal vector
			n = np.zeros([3])
			n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
			n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
			n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
			area = np.linalg.norm(n)
			sum_area += area
		neighbour_code_to_surface_area[code] = sum_area

	# compute the bounding box of the masks to trim
	# the volume to the smallest possible processing subvolume
	mask_all = mask_gt | mask_pred
	bbox_min = np.zeros(3, np.int64)
	bbox_max = np.zeros(3, np.int64)

	# max projection to the x0-axis
	proj_0 = np.max(np.max(mask_all, axis = 2), axis = 1)
	idx_nonzero_0 = np.nonzero(proj_0)[0]
	if len(idx_nonzero_0) == 0:
		return {"distances_gt_to_pred": np.array([]),
		        "distances_pred_to_gt": np.array([]),
		        "surfel_areas_gt": np.array([]),
		        "surfel_areas_pred": np.array([])}

	bbox_min[0] = np.min(idx_nonzero_0)
	bbox_max[0] = np.max(idx_nonzero_0)

	# max projection to the x1-axis
	proj_1 = np.max(np.max(mask_all, axis = 2), axis = 0)
	idx_nonzero_1 = np.nonzero(proj_1)[0]
	bbox_min[1] = np.min(idx_nonzero_1)
	bbox_max[1] = np.max(idx_nonzero_1)

	# max projection to the x2-axis
	proj_2 = np.max(np.max(mask_all, axis = 1), axis = 0)
	idx_nonzero_2 = np.nonzero(proj_2)[0]
	bbox_min[2] = np.min(idx_nonzero_2)
	bbox_max[2] = np.max(idx_nonzero_2)

	print("bounding box min = {}".format(bbox_min))
	print("bounding box max = {}".format(bbox_max))

	# crop the processing subvolume.
	# we need to zeropad the cropped region with 1 voxel at the lower,
	# the right and the back side. This is required to obtain the "full"
	# convolution result with the 2x2x2 kernel
	cropmask_gt = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
	cropmask_pred = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

	cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0] + 1,
	                                bbox_min[1]:bbox_max[1] + 1,
	                                bbox_min[2]:bbox_max[2] + 1]

	cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0] + 1,
	                                  bbox_min[1]:bbox_max[1] + 1,
	                                  bbox_min[2]:bbox_max[2] + 1]

	# compute the neighbour code (local binary pattern) for each voxel
	# the resultsing arrays are spacially shifted by minus half a voxel in each axis.
	# i.e. the points are located at the corners of the original voxels
	kernel = np.array([[[128, 64],
	                    [32, 16]],
	                   [[8, 4],
	                    [2, 1]]])
	neighbour_code_map_gt = scipy.ndimage.filters.correlate(
		cropmask_gt.astype(np.uint8), kernel, mode = "constant", cval = 0)
	neighbour_code_map_pred = scipy.ndimage.filters.correlate(
		cropmask_pred.astype(np.uint8), kernel, mode = "constant", cval = 0)

	# create masks with the surface voxels
	borders_gt = ((neighbour_code_map_gt != 0) &
	              (neighbour_code_map_gt != 255))
	borders_pred = ((neighbour_code_map_pred != 0) &
	                (neighbour_code_map_pred != 255))

	# compute the distance transform (closest distance of each voxel to the surface voxels)
	if borders_gt.any():
		distmap_gt = scipy.ndimage.morphology.distance_transform_edt(
			~borders_gt, sampling = spacing_mm)
	else:
		distmap_gt = np.Inf * np.ones(borders_gt.shape)

	if borders_pred.any():
		distmap_pred = scipy.ndimage.morphology.distance_transform_edt(
			~borders_pred, sampling = spacing_mm)
	else:
		distmap_pred = np.Inf * np.ones(borders_pred.shape)

	# compute the area of each surface element
	surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
	surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

	# create a list of all surface elements with distance and area
	distances_gt_to_pred = distmap_pred[borders_gt]
	distances_pred_to_gt = distmap_gt[borders_pred]
	surfel_areas_gt = surface_area_map_gt[borders_gt]
	surfel_areas_pred = surface_area_map_pred[borders_pred]

	# sort them by distance
	if distances_gt_to_pred.shape != (0,):
		sorted_surfels_gt = np.array(
			sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
		distances_gt_to_pred = sorted_surfels_gt[:, 0]
		surfel_areas_gt = sorted_surfels_gt[:, 1]

	if distances_pred_to_gt.shape != (0,):
		sorted_surfels_pred = np.array(
			sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
		distances_pred_to_gt = sorted_surfels_pred[:, 0]
		surfel_areas_pred = sorted_surfels_pred[:, 1]

	return {"distances_gt_to_pred": distances_gt_to_pred,
	        "distances_pred_to_gt": distances_pred_to_gt,
	        "surfel_areas_gt": surfel_areas_gt,
	        "surfel_areas_pred": surfel_areas_pred}


def compute_average_surface_distance(surface_distances):
	distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
	distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
	surfel_areas_gt = surface_distances["surfel_areas_gt"]
	surfel_areas_pred = surface_distances["surfel_areas_pred"]
	average_distance_gt_to_pred = np.sum(
		distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
	average_distance_pred_to_gt = np.sum(
		distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
	return (average_distance_gt_to_pred, average_distance_pred_to_gt)


def compute_robust_hausdorff(surface_distances, percent):
	distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
	distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
	surfel_areas_gt = surface_distances["surfel_areas_gt"]
	surfel_areas_pred = surface_distances["surfel_areas_pred"]
	if len(distances_gt_to_pred) > 0:
		surfel_areas_cum_gt = np.cumsum(
			surfel_areas_gt) / np.sum(surfel_areas_gt)
		idx = np.searchsorted(surfel_areas_cum_gt, percent / 100.0)
		perc_distance_gt_to_pred = distances_gt_to_pred[min(
			idx, len(distances_gt_to_pred) - 1)]
	else:
		perc_distance_gt_to_pred = np.Inf

	if len(distances_pred_to_gt) > 0:
		surfel_areas_cum_pred = np.cumsum(
			surfel_areas_pred) / np.sum(surfel_areas_pred)
		idx = np.searchsorted(surfel_areas_cum_pred, percent / 100.0)
		perc_distance_pred_to_gt = distances_pred_to_gt[min(
			idx, len(distances_pred_to_gt) - 1)]
	else:
		perc_distance_pred_to_gt = np.Inf

	return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)


def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
	distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
	distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
	surfel_areas_gt = surface_distances["surfel_areas_gt"]
	surfel_areas_pred = surface_distances["surfel_areas_pred"]
	rel_overlap_gt = np.sum(
		surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) / np.sum(surfel_areas_gt)
	rel_overlap_pred = np.sum(
		surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) / np.sum(surfel_areas_pred)
	return (rel_overlap_gt, rel_overlap_pred)


def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
	distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
	distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
	surfel_areas_gt = surface_distances["surfel_areas_gt"]
	surfel_areas_pred = surface_distances["surfel_areas_pred"]
	overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
	overlap_pred = np.sum(
		surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
	surface_dice = (overlap_gt + overlap_pred) / (
			np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
	return surface_dice


def compute_dice_coefficient(mask_gt, mask_pred):
	"""Compute soerensen-dice coefficient.

	compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
	and the predicted mask `mask_pred`.

	Args:
	  mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
	  mask_pred: 3-dim Numpy array of type bool. The predicted mask.

	Returns:
	  the dice coeffcient as float. If both masks are empty, the result is NaN
	"""
	volume_sum = mask_gt.sum() + mask_pred.sum()
	if volume_sum == 0:
		return np.NaN
	volume_intersect = (mask_gt & mask_pred).sum()
	return 2 * volume_intersect / volume_sum


parser = argparse.ArgumentParser(
	description = "Run Performance Metrics.")
parser.add_argument("--gt_folder", default = "",
                    type = str, help = "ground truth data folder")
parser.add_argument("--pred_folder", default = "",
                    type = str, help = "prediction folder")
parser.add_argument("--overlap", default = False,
                    type = bool, help = "apply overlap or not?")
parser.add_argument("--DynUNET", default = "No",
                    type = str, help = "check if it is DynUNET")

args = parser.parse_args()
gt_folder = args.gt_folder
pred_folder = args.pred_folder
overlap = args.overlap

logging.basicConfig(stream = sys.stdout, level = logging.INFO)

spacing_mm = (2, 1, 1)

gt_list = sorted(os.listdir(gt_folder))
pred_list = sorted(os.listdir(pred_folder))

# get the lists of all ground truth and prediction files
gt_list = [gt_folder + file for file in gt_list if file.endswith('_seg.nii.gz')]
pred_list = [pred_folder + file for file in pred_list]

logging.info(f"GT: {gt_list}")
logging.info(f"Pred: {pred_list}")

idx = 0
vd_list = []  # list of volumetric dice scores
sd_list = []  # list of surface dice scores
h100_list = []  # list of hausdorff (100%) scores
h95_list = []  # list of hausdorff (95%) scores

gt = np.array([])
pred = np.array([])

for gt_path, pred_path in zip(gt_list, pred_list):

	logging.info(f"GT: {gt_path}")
	logging.info(f"Pred: {pred_path}")

	load_gt = nib.load(gt_path)
	load_pred = nib.load(pred_path)

	mask_gt = np.array(load_gt.dataobj, dtype = bool)
	mask_pred = np.array(load_pred.dataobj, dtype = bool)

	if args.DynUNET == "No":  # remove the last dimension in the mask for U-Net
		mask_pred = mask_pred[:, :, :, 0]

	print("SHAPE GT ", np.shape(mask_gt), "SHAPE PREDICTION", np.shape(mask_pred))

	if overlap:  # apply the overlap post-processing method
		mask_pred = mask_pred & mask_gt

	surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm)
	print("average surface distance: {} mm".format(compute_average_surface_distance(surface_distances)))

	h100 = compute_robust_hausdorff(surface_distances, 100)
	print("hausdorff (100%):         {} mm".format(h100))

	h95 = compute_robust_hausdorff(surface_distances, 95)
	print("hausdorff (95%):          {} mm".format(h95))

	print("surface overlap at 1mm:   {}".format(compute_surface_overlap_at_tolerance(surface_distances, 1)))

	surface_dice = compute_surface_dice_at_tolerance(surface_distances, 1)
	print("surface dice at 1mm:      {}".format(surface_dice))

	volumetric_dice = compute_dice_coefficient(mask_gt, mask_pred)
	print("volumetric dice:          {}".format(volumetric_dice))

	print("")
	print("expected average_distance_gt_to_pred = 1./6 * 2mm = {}mm".format(1. / 6 * 2))
	print("expected volumetric dice: {}".format(2. * 100 * 100 * 100 / (100 * 100 * 100 + 102 * 100 * 100)))
	print("")

	# Go from N-D array to 1-D array acceptable by sklearn
	mask_gt = mask_gt.ravel()
	mask_pred = mask_pred.ravel()

	gt = np.append(gt, mask_gt)
	pred = np.append(pred, mask_pred)

	if not np.isnan(volumetric_dice):
		vd_list.append(volumetric_dice)
	if not np.isnan(surface_dice):
		sd_list.append(surface_dice)
	if not np.isinf(h100):
		h100_list.append(h100)
	if not np.isinf(h95):
		h95_list.append(h95)

print("Final Average Volumetric Dice - all slices:     {}".format(np.mean(np.array(vd_list))))
print("Final Average Surface Dice - all slices:     {}".format(np.mean(np.array(sd_list))))
print("Final Average Hausdorff 100 - all slices:     {}".format(np.mean(np.array(h100_list))))
print("Final Average Hausdorff 95 - all slices:     {}".format(np.mean(np.array(h95_list))))

fpr, tpr, thresholds = metrics.roc_curve(gt * 1, pred * 1)
auc = metrics.auc(fpr, tpr)

print("AUC: ", auc)
print("FPR: ", fpr)
print("TPR: ", tpr)

TN, FP, FN, TP = metrics.confusion_matrix(gt * 1, pred * 1).ravel()

print("TN, FP, FN, TP: ", TN, FP, FN, TP)

print("Sensitivity: ", TP / (TP + FN))
print("Specificity: ", TN / (TN + FP))
print("Precision: ", TP / (TP + FP))
print("F1: ", 2 * TP / (2 * TP + FP + FN))
