import json
import os
import statistics
from collections import defaultdict
# import plotting
# import matplotlib
from urllib.parse import unquote

import torch


# from IPython.core.debugger import set_trace

# from mpl_toolkits.axes_grid1 import make_axes_locatable

# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
import plotting


def percentile(l, p):
    return l[int(round(p * len(l)))]


category_fns = {
    'animal': '''
    https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Ouchie%20-%20Dang%20That%20Hurt%21%20%28May%202018%29%20_%20FailArmy25.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Get%20Zapped%20-%20Throwback%20Thursday%20%28August%202017%2918.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Jump%20Around%21%20%28March%202018%29%20_%20FailArmy21.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Whip%20It%2C%20Whip%20It%20Real%20Good%20-%20Fails%20of%20the%20Month%20%28November%202017%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Anger%20Management%20-%20Fails%20of%20the%20Week%20%28January%202019%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Get%20Out%20Of%20The%20Way%21%21%20-%20FailArmy%20After%20Dark%20%28Ep.%2012%2960.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Mirror%20Fight%20Back%20-%20Fails%20of%20the%20Week%20%28November%202017%29%20_%20FailArmy37.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Animal%20Fails%20-%20Just%20In%20Time%20For%20Election%202016%20_%20Fail%20Army16.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Confidence%20is%20Key%20-%20Fails%20of%20the%20Week%20%28December%202016%29%20_%20FailArmy4.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Get%20Zapped%20-%20Throwback%20Thursday%20%28August%202017%29112.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2924.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Tourist%20Fails%20%28June%202017%29%20_%20FailArmy10.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20That%27s%20one%20greedy%20cat%2112.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Year%202017%20-%20Part%201%20%28December%202017%29%20_%20FailArmy64.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Classic%20bails%20and%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy42.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%203%20May%202016%20_%20FailArmy19.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Let%27s%20Get%20It%21%21%20-%20FailArmy%20After%20Dark%20%28ep.%202%29195.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hardcore%2C%20Parkour%21%20-%20Fan%20Submissions%20%28May%202018%29%20_%20FailArmy45.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy162.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Mom%27s%20Going%20To%20Be%20Mad%20-%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy0.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Get%20Out%20Of%20The%20Way%21%21%20-%20FailArmy%20After%20Dark%20%28Ep.%2012%29124.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Snow%20Day%20Fails%20-%20It%27s%20Cold%20Out%20There%21%20%28January%202018%29%20_%20FailArmy18.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Mom%27s%20Going%20To%20Be%20Mad%20-%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy9.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Month%20-%20Failing%20into%20Summer%20like...%20%28May%202017%290.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Are%20You%20Serious%21%20-%20Throwback%20Thursday%20%28September%202017%29%20_%20FailArmy57.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hopeless%20Romantic%20-%20Fails%20of%20the%20Week%20%28October%202018%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Am%20Stuck%20In%20This%20Car%21%20-%20Fails%20You%20Missed%20%2316%20_%20FailArmy26.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Break%20Yourself%20-%20Fails%20of%20the%20Week%20%28September%202017%29%20_%20FailArmy39.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Animal%20Fails%20-%20Just%20In%20Time%20For%20Election%202016%20_%20Fail%20Army75.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Get%20Zapped%20-%20Throwback%20Thursday%20%28August%202017%294.mp4
    '''.split('\n'),
    'env': '''
    https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2934.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Mom%27s%20Going%20To%20Be%20Mad%20-%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy59.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%27s%20Got%20Talent%20%28October%202017%29%20_%20FailArmy27.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Sticks%20and%20Stones%20%28July%202018%29%20_%20FailArmy32.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy32.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Carriage%20Carnage%20%28April%202018%29%20_%20FailArmy71.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Drop%20On%20In%21%20-%20Fails%20of%20the%20Week%20%28June%202018%29%20_%20FailArmy14.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Are%20You%20Serious%21%20-%20Throwback%20Thursday%20%28September%202017%29%20_%20FailArmy45.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/You%27ve%20Got%20Bad%20Friends%20-%20Friendship%20Fails%20%28September%202018%29%20_%20FailArmy32.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2946.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hardcore%2C%20Parkour%21%20-%20Fan%20Submissions%20%28May%202018%29%20_%20FailArmy17.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Let%27s%20Do%20This%20Thing%21%20%28April%202017%2943.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2973.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Insult%20to%20Injury%20%28January%202017%29%20_%20FailArmy46.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Tourist%20Fails%20%28June%202017%29%20_%20FailArmy49.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Jump%20Around%21%20%28March%202018%29%20_%20FailArmy61.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/We%27re%20Back%21%20Fails%20of%20the%20Week%20%28May%202019%29%20_%20FailArmy14.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/That%27s%20A%20Sideways%20Wheelie%21%20-%20Throwback%20Thursday%20%28August%202017%29%20_%20FailArmy33.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Tourist%20Fails%20%28June%202017%29%20_%20FailArmy35.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%204%20September%202016%20_%20FailArmy35.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Break%20Yourself%20-%20Fails%20of%20the%20Week%20%28September%202017%29%20_%20FailArmy21.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Party%20On%21%20%20%28Ep.%203%29172.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%27s%20Got%20Talent%20%28October%202017%29%20_%20FailArmy12.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%20-%20Watch%20It%20Drones%21%20%28May%202018%29%20_%20FailArmy8.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20That%27s%20one%20greedy%20cat%2142.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Mom%27s%20Going%20To%20Be%20Mad%20-%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy35.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Sister%20Fails%20_%20Funny%20Sisters%20Fail%20Compilation%20By%20FailArmy%20201639.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20%28March%202017%29%20_%20FailArmy15.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Lookout%20for%20That%20Fence%21%20%28March%202017%29%20_%20FailArmy19.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Treadmill%20Terror%20-%20Fails%20of%20the%20Week%20%28November%202018%29%20_%20FailArmy33.mp4
'''.split('\n'),
    'exec': '''
    https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Mom%27s%20Going%20To%20Be%20Mad%20-%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy34.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2922.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Bee%20Keeper%20Business%20-%20Fails%20of%20the%20Week%20%28November%202018%29%20_%20FailArmy34.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy244.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Get%20Out%20Of%20The%20Way%21%21%20-%20FailArmy%20After%20Dark%20%28Ep.%2012%2971.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%27s%20Got%20Talent%20%28October%202017%29%20_%20FailArmy36.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Party%20On%21%20%20%28Ep.%203%29187.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Bats%20%26%20Balls%20Fail%20Compilation%20_%20By%20FailArmy%20201636.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailFactory%20-%20No%20Pain%2C%20No%20Gain%20%28Workout%20Fails%2913.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Mirror%20Fight%20Back%20-%20Fails%20of%20the%20Week%20%28November%202017%29%20_%20FailArmy8.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%204%20September%202016%20_%20FailArmy9.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Jump%20Around%21%20%28March%202018%29%20_%20FailArmy30.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailFactory%20-%20No%20Pain%2C%20No%20Gain%20%28Workout%20Fails%2914.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%27s%20Got%20Talent%20%28October%202017%29%20_%20FailArmy31.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Insult%20to%20Injury%20%28January%202017%29%20_%20FailArmy10.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20School%20Fails%20Compilation%20_%20%27School%27s%20Out%27%20By%20FailArmy%20201674.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy56.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2925.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Let%27s%20Do%20This%20Thing%21%20%28April%202017%2915.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Are%20You%20Serious%21%20-%20Throwback%20Thursday%20%28September%202017%29%20_%20FailArmy21.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Sticks%20and%20Stones%20%28July%202018%29%20_%20FailArmy5.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20That%27s%20one%20greedy%20cat%2166.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Wheelie%20Gone%20Wrong%20-%20Fails%20of%20the%20Week%20%28May%202018%29%20_%20FailArmy5.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Wheels%20Are%20Coming%20Off%21%20-%20Throwback%20Fails%20%28December%202017%29%20_%20FailArmy15.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Made%20It%21%20-%20Fails%20of%20the%20Week%20%28October%202017%29%20_%20FailArmy20.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%203%20February%202016%20_%20%27I%27m%20OK%2C%20Wheres%20my%20Jetski%21%27FailArmy46.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Ultimate%20Winter%20Fails%20Compilation%20_%20Boards%2C%20Skis%2C%20and%20Snow%20from%20FailArmy18.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Bee%20Keeper%20Business%20-%20Fails%20of%20the%20Week%20%28November%202018%29%20_%20FailArmy33.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%20-%20The%20Eagles%20Are%20Champs%21%21%20%28February%202018%29%20_%20FailArmy13.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy47.mp4
    '''.split('\n'),
    'knowledge': '''
    https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/That%27s%20A%20Sideways%20Wheelie%21%20-%20Throwback%20Thursday%20%28August%202017%29%20_%20FailArmy37.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/34%20Funny%20Kid%20Nominees%20-%20FailArmy%20Hall%20Of%20Fame%20%28May%202017%299.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hilarious%20Science%20Fails%20%28July%202017%29%20_%20FailArmy15.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Sister%20Fails%20_%20Funny%20Sisters%20Fail%20Compilation%20By%20FailArmy%20201612.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hilarious%20Science%20Fails%20%28July%202017%29%20_%20FailArmy5.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Am%20Stuck%20In%20This%20Car%21%20-%20Fails%20You%20Missed%20%2316%20_%20FailArmy101.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Am%20Stuck%20In%20This%20Car%21%20-%20Fails%20You%20Missed%20%2316%20_%20FailArmy39.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Insult%20to%20Injury%20%28January%202017%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy152.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Kid%20Fails%20%28February%202016%29%20_%20FailArmy30.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20That%27s%20one%20greedy%20cat%21102.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hopeless%20Romantic%20-%20Fails%20of%20the%20Week%20%28October%202018%29%20_%20FailArmy29.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%204%20September%202016%20_%20FailArmy4.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hilarious%20Holiday%20Fails%20%28December%202017%29%20_%20FailArmy8.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Be%20Such%20a%20Baby%20-%20Kid%20Fails%20%28September%202018%29%20_%20FailArmy19.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Carriage%20Carnage%20%28April%202018%29%20_%20FailArmy25.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Say%20It%2C%20Don%27t%20Spray%20It%20%28Ep.%207%29109.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Party%20On%21%20%20%28Ep.%203%2962.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hilarious%20Science%20Fails%20%28July%202017%29%20_%20FailArmy23.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Made%20It%21%20-%20Fails%20of%20the%20Week%20%28October%202017%29%20_%20FailArmy18.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Classic%20Redneck%20Waterslide%20-%20Throwback%20Fails%20%28July%202017%2951.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%203%20May%202016%20_%20FailArmy90.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Bait%20and%20Switch%20Fails%20-%20Fooled%20you%21%20%28January%202018%29%20_%20FailArmy9.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Say%20It%2C%20Don%27t%20Spray%20It%20%28Ep.%207%29104.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Get%20Zapped%20-%20Throwback%20Thursday%20%28August%202017%29130.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hilarious%20Science%20Fails%20%28July%202017%29%20_%20FailArmy44.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Classic%20bails%20and%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy114.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Year%202017%20-%20Part%201%20%28December%202017%29%20_%20FailArmy10.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20When%20Enough%20Is%20Enough%21%20%28February%202018%29%20_%20FailArmy33.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Be%20Such%20a%20Baby%20-%20Kid%20Fails%20%28September%202018%29%20_%20FailArmy12.mp4
'''.split('\n'),
    'sensing': '''
    https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%204%20September%202016%20_%20FailArmy9.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Are%20You%20Serious%21%20-%20Throwback%20Thursday%20%28September%202017%29%20_%20FailArmy21.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20That%27s%20one%20greedy%20cat%2166.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy229.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Father%27s%20Day%20Fails%20_%20%27Dad%20Fails%27%20By%20FailArmy%20201612.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%27s%20Got%20Talent%20%28October%202017%29%20_%20FailArmy27.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy255.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Made%20It%21%20-%20Fails%20of%20the%20Week%20%28October%202017%29%20_%20FailArmy29.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy98.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Treadmill%20Terror%20-%20Fails%20of%20the%20Week%20%28November%202018%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Master%20Swordsmen%20-%20Fails%20You%20Missed%20%2317%20_%20FailArmy49.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy102.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hilarious%20Holiday%20Fails%20%28December%202017%29%20_%20FailArmy26.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy179.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20That%27s%20Got%20to%20Sting%20%28Ep.%206%2926.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Lets%20Get%20Stoned%20-%20Rock%20Fails10.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Insult%20to%20Injury%20%28January%202017%29%20_%20FailArmy48.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/You%27ve%20Got%20Bad%20Friends%20-%20Friendship%20Fails%20%28September%202018%29%20_%20FailArmy4.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Wheels%20Are%20Coming%20Off%21%20-%20Throwback%20Fails%20%28December%202017%29%20_%20FailArmy5.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Let%27s%20Do%20This%20Thing%21%20%28April%202017%2913.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Month%20-%20Failing%20into%20Summer%20like...%20%28May%202017%2948.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/It%27s%20All%20Down%20Hill%20From%20Here%20-%20Throwback%20Fails%20%28August%202017%2971.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/You%27ve%20Got%20Bad%20Friends%20-%20Friendship%20Fails%20%28September%202018%29%20_%20FailArmy22.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Crack%20That%20Whip%20-%20Throwback%20Fails%20%28July%202017%2960.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/34%20Funny%20Kid%20Nominees%20-%20FailArmy%20Hall%20Of%20Fame%20%28May%202017%2910.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Let%27s%20Get%20It%21%21%20-%20FailArmy%20After%20Dark%20%28ep.%202%2949.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy169.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Are%20You%20Serious%21%20-%20Throwback%20Thursday%20%28September%202017%29%20_%20FailArmy12.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Say%20It%2C%20Don%27t%20Spray%20It%20%28Ep.%207%2954.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20That%27s%20Going%20To%20Cost%20You%20%20%28Ep.%2010%2938.mp4
    '''.split('\n'),
    'skill': '''
 https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Jump%20Around%21%20%28March%202018%29%20_%20FailArmy30.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailFactory%20-%20No%20Pain%2C%20No%20Gain%20%28Workout%20Fails%2914.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20School%20Fails%20Compilation%20_%20%27School%27s%20Out%27%20By%20FailArmy%20201674.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2925.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Let%27s%20Do%20This%20Thing%21%20%28April%202017%2915.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Wheelie%20Gone%20Wrong%20-%20Fails%20of%20the%20Week%20%28May%202018%29%20_%20FailArmy5.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Bee%20Keeper%20Business%20-%20Fails%20of%20the%20Week%20%28November%202018%29%20_%20FailArmy33.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%20-%20The%20Eagles%20Are%20Champs%21%21%20%28February%202018%29%20_%20FailArmy13.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy47.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailFactory%20-%20No%20Pain%2C%20No%20Gain%20%28Workout%20Fails%2979.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hopeless%20Romantic%20-%20Fails%20of%20the%20Week%20%28October%202018%29%20_%20FailArmy32.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Ultimate%20Parkour%20Fails%20Compilation%20Part%202%20by%20FailArmy%20_%20%27You%20Are%20The%20King%20Of%20Bails%2C%20Man.%2760.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Break%20Yourself%20-%20Fails%20of%20the%20Week%20%28September%202017%29%20_%20FailArmy14.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Lets%20Get%20Stoned%20-%20Rock%20Fails22.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Rapper%20Slips%20off%20the%20Stage%21%21%20-%20Fails%20of%20the%20Week%20%28July%202017%2920.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy142.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Backflip%20Belly%20Flops%20Optional%21%20%28December%202017%29%20_%20FailArmy13.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Saved%20It%21%20%28January%202018%29%20_%20FailArmy8.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Whip%20It%2C%20Whip%20It%20Real%20Good%20-%20Fails%20of%20the%20Month%20%28November%202017%29%20_%20FailArmy18.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Kid%20Fails%20%28February%202016%29%20_%20FailArmy29.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Let%27s%20Get%20It%21%21%20-%20FailArmy%20After%20Dark%20%28ep.%202%29202.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20When%20Enough%20Is%20Enough%21%20%28February%202018%29%20_%20FailArmy19.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20That%27s%20one%20greedy%20cat%216.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailFactory%20-%20No%20Pain%2C%20No%20Gain%20%28Workout%20Fails%2970.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Am%20Stuck%20In%20This%20Car%21%20-%20Fails%20You%20Missed%20%2316%20_%20FailArmy79.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2941.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Are%20You%20Serious%21%20-%20Throwback%20Thursday%20%28September%202017%29%20_%20FailArmy45.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%27s%20Got%20Talent%20%28October%202017%29%20_%20FailArmy29.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Ultimate%20Parkour%20Fails%20Compilation%20Part%202%20by%20FailArmy%20_%20%27You%20Are%20The%20King%20Of%20Bails%2C%20Man.%2715.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Rock%20the%20Boat%20-%20Throwback%20Fails%20%28July%202017%2946.mp4
 '''.split('\n'),
    'multiagent': '''
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Off%20The%20Heezy%20-%20Fails%20of%20the%20Week%20%28August%202018%2928.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/You%27ve%20Got%20Bad%20Friends%20-%20Friendship%20Fails%20%28September%202018%29%20_%20FailArmy18.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Get%20Zapped%20-%20Throwback%20Thursday%20%28August%202017%29126.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Field%20Goal%20Fail%20-%20Fails%20of%20the%20Week%20%28January%202019%299.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Summer%20of%20Fails%20-%20Fails%20of%20the%20Week%20%28July%202018%29%20_%20FailArmy14.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Wheelie%20Gone%20Wrong%20-%20Fails%20of%20the%20Week%20%28May%202018%29%20_%20FailArmy22.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Get%20Zapped%20-%20Throwback%20Thursday%20%28August%202017%2991.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Master%20Swordsmen%20-%20Fails%20You%20Missed%20%2317%20_%20FailArmy87.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Whip%20It%2C%20Whip%20It%20Real%20Good%20-%20Fails%20of%20the%20Month%20%28November%202017%29%20_%20FailArmy32.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Backflip%20Belly%20Flops%20Optional%21%20%28December%202017%29%20_%20FailArmy37.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailFactory%20-%20No%20Pain%2C%20No%20Gain%20%28Workout%20Fails%29129.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Swingset%20Stupidity%20-%20Fails%20of%20the%20Week%20%28May%202019%29%20_%20FailArmy23.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Get%20Out%20Of%20The%20Way%21%21%20-%20FailArmy%20After%20Dark%20%28Ep.%2012%29127.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Say%20It%2C%20Don%27t%20Spray%20It%20%28Ep.%207%299.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/On%20The%20Job%20Fails%20-%20I%20Need%20A%20New%20Job%20%28October%202017%29%20_%20FailArmy37.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Workout%20Fails%20-%20New%20Year%2C%20Same%20Me%20%28January%202017%29%20_%20FailArmy24.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Everybody%20Sing%21%21%20%28February%202018%29%20_%20FailArmy28.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Caught%20Slippin%27%20-%20Fails%20of%20the%20Week%20%28February%202019%29%20_%20FailArmy15.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/It%27s%20All%20Down%20Hill%20From%20Here%20-%20Throwback%20Fails%20%28August%202017%2979.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy89.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Drop%20On%20In%21%20-%20Fails%20of%20the%20Week%20%28June%202018%29%20_%20FailArmy0.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Carriage%20Carnage%20%28April%202018%29%20_%20FailArmy67.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/34%20Funny%20Kid%20Nominees%20-%20FailArmy%20Hall%20Of%20Fame%20%28May%202017%2911.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Backflip%20Belly%20Flops%20Optional%21%20%28December%202017%29%20_%20FailArmy70.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hardcore%2C%20Parkour%21%20-%20Fan%20Submissions%20%28May%202018%29%20_%20FailArmy5.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Ninja%20Fails%20-%20Sweep%20The%20Leg%21%20%28March%202017%29%20_%20FailArmy15.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Classic%20Redneck%20Waterslide%20-%20Throwback%20Fails%20%28July%202017%2971.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/You%27ve%20Got%20Bad%20Friends%20-%20Friendship%20Fails%20%28September%202018%29%20_%20FailArmy24.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Let%27s%20Get%20It%21%21%20-%20FailArmy%20After%20Dark%20%28ep.%202%29178.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Wheels%20Are%20Coming%20Off%21%20-%20Throwback%20Fails%20%28December%202017%29%20_%20FailArmy3.mp4
'''.split('\n'),
    'plan_err': '''
    https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Summer%20of%20Fails%20-%20Fails%20of%20the%20Week%20%28July%202018%29%20_%20FailArmy26.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/It%27s%20All%20Down%20Hill%20From%20Here%20-%20Throwback%20Fails%20%28August%202017%2970.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Sister%20Fails%20_%20Funny%20Sisters%20Fail%20Compilation%20By%20FailArmy%2020167.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Whip%20It%2C%20Whip%20It%20Real%20Good%20-%20Fails%20of%20the%20Month%20%28November%202017%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/That%27s%20A%20Sideways%20Wheelie%21%20-%20Throwback%20Thursday%20%28August%202017%29%20_%20FailArmy5.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Expensive%20Fails%20-%20That%27s%20Going%20To%20Cost%20You%21%20%28July%202017%295.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20That%27s%20Got%20to%20Sting%20%28Ep.%206%29124.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Field%20Goal%20Fail%20-%20Fails%20of%20the%20Week%20%28January%202019%2933.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Get%20Out%20Of%20The%20Way%21%21%20-%20FailArmy%20After%20Dark%20%28Ep.%2012%2969.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Tourist%20Fails%20%28June%202017%29%20_%20FailArmy0.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/34%20Funny%20Kid%20Nominees%20-%20FailArmy%20Hall%20Of%20Fame%20%28May%202017%290.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20You%20Missed%20-%20Not%20the%20Bees%20%28April%202018%29%20_%20Failarmy69.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20That%27s%20one%20greedy%20cat%211.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Drop%20On%20In%21%20-%20Fails%20of%20the%20Week%20%28June%202018%29%20_%20FailArmy26.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Anger%20Management%20-%20Fails%20of%20the%20Week%20%28January%202019%29%20_%20FailArmy0.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Anger%20Management%20-%20Fails%20of%20the%20Week%20%28January%202019%29%20_%20FailArmy7.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20That%27s%20Got%20to%20Sting%20%28Ep.%206%2928.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20School%20Fails%20Compilation%20_%20%27School%27s%20Out%27%20By%20FailArmy%20201659.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Carriage%20Carnage%20%28April%202018%29%20_%20FailArmy38.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Kid%20Fails%20%28February%202016%29%20_%20FailArmy12.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20You%20Missed%20-%20Not%20the%20Bees%20%28April%202018%29%20_%20Failarmy55.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Off%20The%20Heezy%20-%20Fails%20of%20the%20Week%20%28August%202018%2939.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Does%20That%20Sheep%20Have%20Sunglasses%20%20%28Ep.%208%2951.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%204%20September%202016%20_%20FailArmy8.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Off%20The%20Heezy%20-%20Fails%20of%20the%20Week%20%28August%202018%2912.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Let%27s%20Get%20It%21%21%20-%20FailArmy%20After%20Dark%20%28ep.%202%29215.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%203%20February%202016%20_%20%27I%27m%20OK%2C%20Wheres%20my%20Jetski%21%27FailArmy49.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Carriage%20Carnage%20%28April%202018%29%20_%20FailArmy28.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Year%202017%20-%20Part%201%20%28December%202017%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%202%20March%202016%20_%20%27I%20Think%20We%20Got%20What%20We%27re%20Looking%20For%27%20by%20FailArmy0.mp4
    '''.split('\n'),
    'singleagent': '''
    https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailFactory%20-%20No%20Pain%2C%20No%20Gain%20%28Workout%20Fails%2921.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Month%20%28December%202016%29%20_%20FailArmy49.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Am%20Stuck%20In%20This%20Car%21%20-%20Fails%20You%20Missed%20%2316%20_%20FailArmy0.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hilarious%20Holiday%20Fails%20%28December%202017%29%20_%20FailArmy6.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Get%20Out%20Of%20The%20Way%21%21%20-%20FailArmy%20After%20Dark%20%28Ep.%2012%2992.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Sticks%20and%20Stones%20%28July%202018%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Carriage%20Carnage%20%28April%202018%29%20_%20FailArmy20.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Are%20You%20Serious%21%20-%20Throwback%20Thursday%20%28September%202017%29%20_%20FailArmy68.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hilarious%20Science%20Fails%20%28July%202017%29%20_%20FailArmy44.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Lookout%20for%20That%20Fence%21%20%28March%202017%29%20_%20FailArmy24.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Sticks%20and%20Stones%20%28July%202018%29%20_%20FailArmy11.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%27s%20Got%20Talent%20%28October%202017%29%20_%20FailArmy7.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20%28March%202017%29%20_%20FailArmy54.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Get%20Out%20Of%20The%20Way%21%21%20-%20FailArmy%20After%20Dark%20%28Ep.%2012%2943.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/That%27s%20A%20Sideways%20Wheelie%21%20-%20Throwback%20Thursday%20%28August%202017%29%20_%20FailArmy86.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Let%27s%20Do%20This%20Thing%21%20%28April%202017%298.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Say%20It%2C%20Don%27t%20Spray%20It%20%28Ep.%207%297.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Ouchie%20-%20Dang%20That%20Hurt%21%20%28May%202018%29%20_%20FailArmy6.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/It%27s%20All%20Down%20Hill%20From%20Here%20-%20Throwback%20Fails%20%28August%202017%2993.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Classic%20bails%20and%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy114.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Guess%20it%27s%20Time%20to%20Leave%20-%20Throwback%20Thursday%20%28October%202017%29%20_%20FailArmy34.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/It%27s%20All%20Down%20Hill%20From%20Here%20-%20Throwback%20Fails%20%28August%202017%2942.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Let%27s%20Get%20It%21%21%20-%20FailArmy%20After%20Dark%20%28ep.%202%2997.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Say%20It%2C%20Don%27t%20Spray%20It%20%28Ep.%207%2983.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Let%27s%20Get%20It%21%21%20-%20FailArmy%20After%20Dark%20%28ep.%202%2946.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Best%20Fails%20of%20the%20Week%203%20May%202016%20_%20FailArmy65.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Hopeless%20Romantic%20-%20Fails%20of%20the%20Week%20%28October%202018%29%20_%20FailArmy16.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Master%20Swordsmen%20-%20Fails%20You%20Missed%20%2317%20_%20FailArmy5.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20%28March%202017%29%20_%20FailArmy28.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/The%20Wheels%20Are%20Coming%20Off%21%20-%20Throwback%20Fails%20%28December%202017%29%20_%20FailArmy20.mp4
    '''.split('\n'),
    'unexpected': '''
    https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Throwback%20Fails%20-%20Jump%20Around%21%20%28March%202018%29%20_%20FailArmy37.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Crack%20That%20Whip%20-%20Throwback%20Fails%20%28July%202017%299.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/I%20Made%20It%21%20-%20Fails%20of%20the%20Week%20%28October%202017%29%20_%20FailArmy36.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Party%20On%21%20%20%28Ep.%203%29180.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Whip%20It%2C%20Whip%20It%20Real%20Good%20-%20Fails%20of%20the%20Month%20%28November%202017%29%20_%20FailArmy2.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20School%20Fails%20Compilation%20_%20%27School%27s%20Out%27%20By%20FailArmy%20201619.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Sister%20Fails%20_%20Funny%20Sisters%20Fail%20Compilation%20By%20FailArmy%2020160.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20That%27s%20Going%20To%20Cost%20You%20%20%28Ep.%2010%29118.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Rapper%20Slips%20off%20the%20Stage%21%21%20-%20Fails%20of%20the%20Week%20%28July%202017%2915.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Insult%20to%20Injury%20%28January%202017%29%20_%20FailArmy48.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Classic%20bails%20and%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy0.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Rapper%20Slips%20off%20the%20Stage%21%21%20-%20Fails%20of%20the%20Week%20%28July%202017%2937.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20You%20Missed%20-%20Not%20the%20Bees%20%28April%202018%29%20_%20Failarmy104.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Party%20On%21%20%20%28Ep.%203%2994.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Party%20On%21%20%20%28Ep.%203%29160.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy162.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Don%27t%20Get%20Zapped%20-%20Throwback%20Thursday%20%28August%202017%295.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Party%20On%21%20%20%28Ep.%203%29169.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Does%20That%20Sheep%20Have%20Sunglasses%20%20%28Ep.%208%2931.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20After%20Dark%20-%20Say%20It%2C%20Don%27t%20Spray%20It%20%28Ep.%207%2958.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Month%20-%20Failing%20into%20Summer%20like...%20%28May%202017%290.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Workout%20Fails%20-%20New%20Year%2C%20Same%20Me%20%28January%202017%29%20_%20FailArmy9.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Bats%20%26%20Balls%20Fail%20Compilation%20_%20By%20FailArmy%20201647.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/FailArmy%20Playlist%20-%20Inflatables%20and%20Big%20Ballers%20%28March%202019%29%20_%20FailArmy188.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Week%20-%20Let%27s%20Do%20This%20Thing%21%20%28April%202017%2927.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Funny%20Tourist%20Fails%20%28June%202017%29%20_%20FailArmy16.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Let%27s%20Get%20It%21%21%20-%20FailArmy%20After%20Dark%20%28ep.%202%29133.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Fails%20of%20the%20Month%20%28December%202016%29%20_%20FailArmy3.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Classic%20bails%20and%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy97.mp4
https://cv.cs.columbia.edu/dave/fails/viz_gif/scenes/val/val/Classic%20bails%20and%20Throwback%20Fails%20%28June%202017%29%20_%20FailArmy33.mp4
    '''.split('\n')
}

withheld_actions = ['base jumping', 'bench pressing', 'bobsledding', 'bodysurfing', 'bowling',
    'catching or throwing baseball', 'catching or throwing frisbee', 'catching or throwing softball',
    'curling (sport)', 'deadlifting', 'disc golfing', 'dodgeball', 'dribbling basketball',
    'dunking basketball', 'exercising with an exercise ball',
    'fencing (sport)', 'golf chipping', 'golf driving', 'golf putting', 'hitting baseball',
    'hockey stop',
    'hurling (sport)', 'javelin throw', 'juggling soccer ball', 'kicking field goal',
    'kicking soccer ball', 'longboarding', 'luge', 'passing American football (in game)',
    'passing american football (not in game)', 'passing soccer ball',
    'playing badminton', 'playing basketball', 'playing cricket', 'playing field hockey',
    'playing ice hockey',
    'playing kickball', 'playing netball', 'playing paintball', 'playing ping pong',
    'playing squash or racquetball',
    'playing tennis', 'pole vault', 'pull ups', 'push up', 'shooting goal (soccer)',
    'snatch weight lifting', 'swinging baseball bat',
    'throwing ball (not baseball or American football)', 'throwing discus', ]

if __name__ == "__main__":
    with open(
            'JSON PATH HERE') as f:
        data = json.load(f)

    # fns = glob(
    #     '/local/vondrick/dave/slidingwindow/fails_kinetics_features/fails_kinetics_features_*.json')
    #
    # kin_cls = torch.load('/local/vondrick/dave/fails/kinetics_classes.pt')
    #
    # kindata = {}
    # for k, v in kinetics.process().items():
    #     kindata[os.path.basename(k)] = v

    anns_path = 'PATH TO/all_mturk_data.json'
    human_anns_path = 'PATH TO/fourth/all_mturk_data.json'
    fps = 16

    fndata = defaultdict(list)

    confusion = torch.zeros((3, 3))  # confusion[gt][pred]

    for d in data:
        fn = os.path.basename(d['fn'])
        fn = os.path.splitext(fn)[0]
        fndata[fn].append(d)
        confusion[d['y']][d['y_hat']] += 1

    for fn in fndata:
        fndata[fn] = sorted(fndata[fn], key=lambda e: e['t_start'])

    with open(anns_path) as f:
        anns = json.load(f)

    with open(human_anns_path) as f:
        human_anns = json.load(f)

    plotting.FigureBuilder(anns, confusion=confusion, ok_names=['kinetics_dist', 'places_dist'])()

    errs = []
    rel_errs = []
    accs = []

    act_errs = defaultdict(list)
    act_rel_errs = defaultdict(list)
    act_accs = defaultdict(list)

    # acc = []
    # for ann_data in anns.values():
    #     if fn not in fndata:
    #         continue
    #     fail_t_gt = statistics.median(ann_data['t'])
    #     if not 0.01 <= statistics.median(ann_data['rel_t']) <= 0.99:
    #         continue
    #     if fail_t_gt < 0:
    #         continue
    #     clip_end = 1
    #     pred_t = human_anns[fn]['t'][0]
    #     gt_t = statistics.median(ann_data['t'])
    #     n = 0
    #     while clip_end < ann_data['len']:
    #         if clip_end < pred_t:  # pred = 0
    #             acc.append(clip_end < gt_t)
    #         elif clip_end - 1 > pred_t:  # pred = 2
    #             acc.append(clip_end - 1 > gt_t)
    #         else:  # pred = 1
    #             acc.append(clip_end - 1 < gt_t < clip_end)
    #         clip_end += 3
    #         n -= 1
    #         if not n: break
    # print(statistics.mean(acc))

    catk = list(category_fns.keys())[9]

    filter_fns = category_fns[catk]


    filter_fns = list(
        filter(lambda x: x, map(lambda url: os.path.splitext(os.path.basename(unquote(url).strip()))[0], filter_fns)))

    filter_fns = None

    if filter_fns: print(catk)

    filter_fns_reached = 0

    # stdev_data = []
    fn_errs = []

    preds = defaultdict(int)

    usemedian = False

    for fn, ann_data in anns.items():
        if fn not in fndata or (filter_fns is not None and fn not in filter_fns):
            continue
        if filter_fns is not None:
            filter_fns_reached += 1
        fail_t_gt = statistics.median(ann_data['t'])
        if not 0.01 <= statistics.median(ann_data['rel_t']) <= 0.99:
            continue
        if fail_t_gt < 0:
            continue
        data = fndata[fn]
        # # RANDOM - COMMENT OUT IF NOT TESTING RANDOM
        # t_f = statistics.median(ann_data['rel_t'])
        # E_err = (2 * t_f * t_f - 2 * t_f + 1) / 2
        # rel_errs.append(E_err)
        # errs.append(E_err * ann_data['len'])
        # at_k = .25
        # accs.append((min(ann_data['len'], fail_t_gt + at_k) - max(0, fail_t_gt - at_k)) / ann_data['len'])
        # continue  # RANDOM ONLY
        # for d in data:
        #     t_d = (d['t_start'] + d['t_end']) / 2
        #     # if any(t_d < (t - 0.5) for t in ann_data['t']):
        #     # if t_d < fail_t_gt:
        #     if -1 < d['y'] < 2:
        #         p0, p1, p2 = d['y_hat_vec']
        #         pred_fail = p1 > p0
        #         gt_fail = max(abs(t_d + 1.5 - t) < 0.5 for t in [statistics.median(ann_data['t'])])
        #         preds[(pred_fail, gt_fail)] += 1
        #     p0, p1, p2 = d['y_hat_vec']
        #     # if d['y'] == 0:
        #     #     if d['y_hat'] == 0:
        #     #         preds['tn'] += 1
        #     #     else:
        #     #         preds['fp'] += 1
        #     # elif d['y'] == 1:
        #     #     if d['y_hat'] == 1:
        #     #         preds['tp'] += 1
        #     #     else:
        #     #         preds['fn'] += 1
        #     # else:
        #     #     break
        #     # y_hat = 1
        #     # if t_d < (fail_t_gt-.5): y_hat = 0
        #     # elif t_d > fail_t_gt+.5: y_hat = 2
        #     # preds['tp'] += int(y_hat == d['y'])
        #     # preds['fp'] += int(y_hat != d['y'])
        # continue
        max_el = max(data, key=lambda el: el['y_hat_vec'][1])
        # fail_t_pred = human_anns[fn]['t'][0]
        fail_t_pred = (max_el['t_start'] + max_el['t_end']) / 2
        # fail_t_pred = random.uniform(0, ann_data['len'])
        # fail_t_pred = ann_data['len'] * 0.5
        # min_el = max(data, key=lambda el: el['y_hat_vec'][0])
        # nonfail_t_pred = (min_el['t_start'] + min_el['t_end']) / 2
        # acc_cls = 100 * int(fail_t_pred >= fail_t_gt)
        if usemedian:
            ann_data['t'] = [statistics.median(ann_data['t'])]
        acc_cls = 100 * (min(abs(fail_t_pred - t) for t in ann_data['t']) <= .25)
        err_t = min(abs(fail_t_pred - t) for t in ann_data['t'])
        rel_err_t = 100 * err_t / ann_data['len']
        fn_errs.append((fn, rel_err_t))
        # stdev_data.append((ann_data['rel_stdev'], rel_err_t))
        errs.append(err_t)
        rel_errs.append(rel_err_t)
        accs.append(acc_cls)
        # act_errs[kindata[fn]].append(err_t)
        # act_rel_errs[kindata[fn]].append(rel_err_t)
        # act_accs[kindata[fn]].append(acc_cls)
    # stdev_errs_hist = defaultdict(list)
    # for s, e in stdev_data:
    #     stdev_errs_hist[s < 0.33].append(e)
    # for k, v in stdev_errs_hist.items():
    #     stdev_errs_hist[k] = (statistics.mean(v), statistics.stdev(v))
    # precision = 100 * preds[(True, True)] / (preds[(True, True)] + preds[(True, False)])
    # recall = 100 * preds[(True, True)] / (preds[(True, True)] + preds[(False, True)])
    # print(f'precision: {precision}')
    # print(f'recall: {recall}')
    # print(f'f1: {2 * precision * recall / (precision + recall)}')
    # print(
    #     f'accuracy: {100 * (preds[(True, True)] + preds[(False, False)]) / (preds[(True, True)] + preds[(False, True)] + preds[(False, False)] + preds[(True, False)])}')
    # print(f'two way accuracy: { 100 * (preds["tp"]+preds["tn"])/(preds["tp"]+preds["tn"]+preds["fp"]+preds["fn"])}')
    for l in [act_errs, act_rel_errs, act_accs]:
        for k, v in l.items():
            l[k] = (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0)

    # print(
    #     f'most classifiable: {", ".join([kin_cls[_] for _, __ in Counter({k: v[0] for k, v in act_accs.items()}).most_common(10)])}')
    # print(
    #     f'least classifiable: {", ".join([kin_cls[_] for _, __ in Counter({k: -v[0] for k, v in act_accs.items()}).most_common(10)])}')
    # print(
    #     f'most localizable: {", ".join([kin_cls[_] for _, __ in Counter({k: v[0] for k, v in act_rel_errs.items()}).most_common(10)])}')
    # print(
    #     f'least localizable: {", ".join([kin_cls[_] for _, __ in Counter({k: -v[0] for k, v in act_rel_errs.items()}).most_common(10)])}')
    if filter_fns:
        print(f'evaluated {filter_fns_reached} clips')

    print(
        'mean abs error: {0:4} +- {1:4}'.format(
            statistics.mean(errs), statistics.stdev(errs)),
        '\nmean rel error: {0:4} +- {1:4}'.format(
            statistics.mean(rel_errs), statistics.stdev(rel_errs)),
        '\nmean accuracy: {0:4} +- {1:4}'.format(
            statistics.mean(accs), statistics.stdev(accs)))
    print(
        'median abs error: {0:4} +- {1:4}'.format(
            statistics.median(errs), statistics.stdev(errs)),
        '\nmedian rel error: {0:4} +- {1:4}'.format(
            statistics.median(rel_errs), statistics.stdev(rel_errs)),
        '\nmedian accuracy: {0:4} +- {1:4}'.format(
            statistics.median(accs), statistics.stdev(accs)))

    pass
