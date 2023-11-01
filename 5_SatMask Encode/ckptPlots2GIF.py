import glob
import imageio.v2 as imageio

RESULTS_DIR = './results/Conv/'

anim_file = RESULTS_DIR + 'trainingProgress.gif'

with imageio.get_writer(anim_file, mode='I', fps=2) as writer:
    filenames = glob.glob(RESULTS_DIR + 'ckptResults/imageckpt_*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)