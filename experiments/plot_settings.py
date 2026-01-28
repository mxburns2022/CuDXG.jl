import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif']= 'Times New Roman'
mpl.rcParams['mathtext.fontset']= 'stix'
mpl.rcParams['font.size']= 20
mpl.rcParams["figure.dpi"] = 300
# plt.rc('pdf',fonttype = 1)
mpl.rcParams['text.usetex'] = True #Let TeX do the typsetting
mpl.rcParams['text.latex.preamble'] = '\\usepackage{sansmath}\n\\sansmath' #Force sans-serif math mode (for axes labels)
mpl.rcParams['font.family'] = 'sans-serif' # ... for regular text
mpl.rcParams['font.sans-serif'] = 'Helvetica' # Choose a nice font here
