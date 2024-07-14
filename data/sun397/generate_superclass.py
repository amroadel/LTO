import os
import json
import collections

import numpy as np
from .prompts import classes as sun397_classes

def main():
    superclass_names = [
        'shopping and dining',
        'workplace (office building, factory, lab, etc.)',
        'home or hotel',
        'transportation (vehicle interiors, stations, etc.)',
        'sports and leisure',
        'cultural (art, education, religion, etc.)',
        'water, ice, snow',
        'mountains, hills, desert, sky',
        'forest, field, jungle',
        'transportation (roads, parking, bridges, boats, airports, etc.)',
        'cultural or historical building(place)',
        'sports fields, parks, leisure spaces',
        'industrial and construction',
        'houses, cabins, gardens, and farms',
        'commercial buildings, shops, markets, cities, and towns',
    ]

    hierachy_fpath = os.path.join(
        os.path.dirname(__file__), 'hierachy.csv')
    x = np.loadtxt(hierachy_fpath, dtype=str)
    
    man_made_element_id = 9 # exclude man-made-element

    superclass_dict = collections.defaultdict(list)
    for i, tokens in enumerate(x):
        c = tokens[0]
        labels = tokens[4:].astype(int)
        
        # Exclude mem-made element attributes
        labels = list(labels[:man_made_element_id]) + list(labels[man_made_element_id+1:])

        superclass_id = labels.index(1)
        superclass_dict[superclass_names[superclass_id]].append(
            sun397_classes[i])
        
    superclass_dict = { key: superclass_dict[key] for key in sorted(superclass_dict.keys())}

    with open(os.path.join(
        os.path.dirname(__file__), 
        f'superclass-{len(superclass_names)}.json'), 
    'w') as f:
        json.dump(superclass_dict, f, indent=2)


if __name__ == '__main__':
    main()


