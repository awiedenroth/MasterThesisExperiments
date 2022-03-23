from typing import List, Dict, Any

import numpy as np

def calculate_average(ergebnisse):
    agg_dict = aggregate_dicts(ergebnisse)
    average = reduce_agg_dict(agg_dict)
    return average

def calculate_conf_average(ergebnisse:List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    list_of_first = list_of_firsts(ergebnisse)
    result = []
    for list in list_of_first:
        result.append(calculate_average(list))
    return result

def calculate_average_report(ergebnisse:List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    agg1 = aggregate_dicts(ergebnisse)
    for key, liste in agg1.items():
        agg2 = aggregate_dicts(liste)
        reduced = reduce_agg_dict(agg2)
        agg1[key] = reduced
    return agg1

    # ich iteriere durch die dicts für jeden k durchgang und summiere jeweils die werte an den richtigen positionen
    # dann teile ich durch k
    # damit bekomme ich den average metrik für die configuration

def aggregate_dicts(dict_list: List[Dict[str, Any]])-> Dict[str, List[Any]]:
    return {key: [i[key] for i in dict_list] for key in dict_list[0]}

def reduce_agg_dict(agg_dict: Dict[str, List[Any]], func=np.mean) -> Dict[str, Any]:
    return {key: func(agg_dict[key]) for key in agg_dict}

def list_of_firsts(ergebnisse:List[List[Dict[str, Any]]])->List[List[Dict[str, Any]]]:
    result = []
    for i in range(len(ergebnisse)):
        zwischen = []
        for list in ergebnisse:
            zwischen.append(list[i])
        result.append(zwischen)

    return result


