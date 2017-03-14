# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

from PIL import Image
import random

WIDTH = 161
HEIGHT = 100
RAW_COUNT = 32

BRANDING_BAR_PATH = "../../framework_tvb/tvb/core/services/resources/branding_bar.png"



def glue_2_images(image1, image2, final_path):
    final_image = Image.new("RGBA", (2 * WIDTH, HEIGHT))

    img = Image.open(image1)
    final_image.paste(img, (0, 0), img)
    img = Image.open(image2)
    final_image.paste(img, (WIDTH, 0), img)

    final_image.save(final_path, "PNG")
    print "Saved image:", final_path



def glue_images(no_of_rows, input_img_prefix, target_img_prefix, landscape=True):
    no_of_columns = no_of_rows if landscape else no_of_rows / 4
    final_image = Image.new("RGBA", (2 * WIDTH * no_of_columns, HEIGHT * no_of_rows))
    raw_images = range(RAW_COUNT)
    random.shuffle(raw_images)
    raw_idx = 0

    for column in xrange(no_of_rows if landscape else no_of_rows / 4):

        for row in xrange(no_of_rows):
            if raw_idx >= RAW_COUNT:
                random.shuffle(raw_images)
                raw_idx = 0

            image_path = input_img_prefix + str(raw_images[raw_idx]) + ".png"
            img = Image.open(image_path)
            final_image.paste(img, (column * 2 * WIDTH, row * HEIGHT), img)
            raw_idx += 1

    branding_bar = Image.open(BRANDING_BAR_PATH)
    final_image.paste(branding_bar, (0, final_image.size[1] - branding_bar.size[1]), branding_bar)

    final_path = target_img_prefix + str(no_of_rows) + ".png"
    final_image.save(final_path, "PNG")

    print "Saved image:", final_path



if __name__ == "__main__":
    for i in xrange(RAW_COUNT):
        left_view = "raw/snapshot-" + str(i) + "A.png"
        right_view = "raw/snapshot-" + str(i) + "B.png"
        result = "glued/brain-" + str(i) + ".png"
        glue_2_images(left_view, right_view, result)

    glue_images(8, "glued/brain-", "cohort-P-", False)
    glue_images(16, "glued/brain-", "cohort-P-", False)
    glue_images(32, "glued/brain-", "cohort-P-", False)
    glue_images(64, "glued/brain-", "cohort-P-", False)
    glue_images(128, "glued/brain-", "cohort-P-", False)
    glue_images(8, "glued/brain-", "cohort-")
    glue_images(16, "glued/brain-", "cohort-")
    glue_images(32, "glued/brain-", "cohort-")
    glue_images(64, "glued/brain-", "cohort-")
    glue_images(128, "glued/brain-", "cohort-")
