import numpy as np
import sexpdata
import sys

def parse_tree(p):
    if "'" in p:
        p = "none"
    parsed = sexpdata.loads(p)
    extracted = extract_parse(parsed)
    return extracted

def extract_parse(p):
    if isinstance(p, sexpdata.Symbol):
        return p.value()
    elif isinstance(p, int):
        return str(p)
    elif isinstance(p, bool):
        return str(p).lower()
    elif isinstance(p, float):
        return str(p).lower()
    return tuple(extract_parse(q) for q in p)

def parse_to_layout(parse):
    if isinstance(parse, str):
        return ('Find', parse)

    head = parse[0]
    children = parse[1:]

    conjunctions = ["and"]
    if head in conjunctions:
        label_head = ('And', 'and')
        labels_below = tuple(parse_to_layout(child) for child in children )
        labels_here = (label_head, labels_below)
    
    transforming_adjectives = ["in", "on", "above", "below", "besides", "over", "beside", "through", "inside", "out_of", "at", "with", "behind", "on_top_of", "next_to", "outside_of", "into", "between", "around", "toward", "towards", "under", "within", "in_front_of", "near", "nearest", "without", "because_of", "like", "along", "about", "against"]
    elif head in transforming_adjectives:
        label_head = ('Transform', head)
        labels_below = tuple(parse_to_layout(child) for child in children )
        labels_here = (label_head,) + labels_below    
    else:
        label_head = ('Describe', head)
        labels_below = tuple(parse_to_layout(child) for child in children )
        labels_here = (label_head,) + labels_below

    return labels_here


if __name__ == "__main__":

    all_layouts = []
    with open(sys.argv[1]) as f:
        for line in f.readlines():
            parse_group = line.strip()
            if parse_group != "":
                parse_strs = parse_group.split(";")
                parses = [parse_tree(p) for p in parse_strs]
                parses = [("what", "thing") if p == "none" else p for p in parses]
                layouts = [parse_to_layout(parse) for parse in parses]

                all_layouts.append(layouts)

    with open(sys.argv[2], "w") as f:
        f.write( "\n".join(map(str,all_layouts)) )

