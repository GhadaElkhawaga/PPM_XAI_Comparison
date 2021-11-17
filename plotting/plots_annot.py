import matplotlib.pyplot as plt


def plot(x,y,title_str,cls,marker):
  plt.plot(x, y, label = "%s_%s" %(title_str,cls),marker=marker,markersize=5)
  return


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points", ha='center', va='bottom', size=6)
    return




