import json
import os
from lib.parse_tree_to_primary_layout import *
from lib.primary_to_module_layout import *

# set_names = ['train2014', 'test2015', 'val2014', 'test-dev2015']
set_names = ['train2014-sub', 'val2014-sub', 'test2014-sub']

root_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)),  '..')
layout_data_dir = os.path.join(root_dir, 'preprocessed_data/layouts')


raw_questions_dir = os.path.join(layout_data_dir, 'raw_questions')
if not os.path.exists( raw_questions_dir ):
    os.makedirs(raw_questions_dir)

question_ids_dir = os.path.join(layout_data_dir, 'question_ids')
if not os.path.exists(question_ids_dir): 
    os.makedirs(question_ids_dir)

question_parse_trees_dir = os.path.join(layout_data_dir, 'question_parse_trees')
if not os.path.exists(question_parse_trees_dir): 
    os.makedirs(question_parse_trees_dir)

primary_layouts_dir = os.path.join(layout_data_dir, 'primary_layouts')
if not os.path.exists(primary_layouts_dir): 
    os.makedirs(primary_layouts_dir)

module_layouts_dir = os.path.join(layout_data_dir, 'module_layouts')
if not os.path.exists(module_layouts_dir): 
    os.makedirs(module_layouts_dir)



# Dump Raw Questions Files here
print "Dumping Raw Questions in this Directory..."
for set_name in set_names:
    print "{} ...".format(set_name)
    question_file = os.path.join(root_dir, 
        'raw_data/Questions/v2_OpenEnded_mscoco_{}_questions.json'.format(set_name))

    question_ids_file = os.path.join(question_ids_dir, 
            'questions_{}.txt'.format(set_name)  )

    raw_question_file = os.path.join(layout_data_dir, 
        'raw_questions/questions_{}.txt'.format(set_name)  )

    with open(question_file) as f:
        questions = json.load(f)["questions"]

    with open(question_ids_file, 'w') as f:
        for question in questions:
            f.write( str(question["question_id"]) + "\n" )

    with open(raw_question_file, 'w') as f:
        for question in questions:
            f.write( question["question"].strip() + "\n" )

# Get their Parse Trees
print "Building Parse Trees for Questions ..."

for set_name in set_names:
    print "{} ...".format(set_name)
    raw_question_file = raw_question_file = os.path.join(layout_data_dir, 
                        'raw_questions/questions_{}.txt'.format(set_name)  )

    question_parse_tree_file = os.path.join( question_parse_trees_dir, 
                        'questions_parse_trees_{}.txt'.format(set_name) )

    command = '''
    java -mx150m -cp "$STANFORDPARSER/libexec/*" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
        -outputFormat "words,typedDependencies" -outputFormatOptions "stem,collapsedDependencies,includeTags" \
        -sentences newline \
        edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
        {} \
        > {}
    '''.format(raw_question_file, question_parse_tree_file)
    
    os.system(command)


# Get their Primary Layouts
print "Building Primary Layouts ofr Questions ..."

for set_name in set_names:
    print "{} ...".format(set_name)

    question_parse_tree_file = os.path.join( question_parse_trees_dir, 
                        'questions_parse_trees_{}.txt'.format(set_name) )

    primary_layout_file = os.path.join( primary_layouts_dir, 
                        'questions_primary_layouts_{}.txt'.format(set_name) )

    # For ReAttend Not Available
    #parser = LfParser(use_relations=False, max_conjuncts=2, max_leaves=2)

    # For ReAttend Available
    parser = LfParser(use_relations=True, max_conjuncts=2, max_leaves=4)

    output_lines = []
    with open(question_parse_tree_file) as f:
        for parses in parser.parse_all( f ):
            output_lines.append(";".join(parses))

    with open( primary_layout_file, "w") as f:
        f.write( "\n".join(output_lines) )


# Get their Module Layouts
print "Building Module Layouts of Questions ..."
for set_name in set_names:
    print "{} ...".format(set_name)

    primary_layout_file = os.path.join( primary_layouts_dir, 
                        'questions_primary_layouts_{}.txt'.format(set_name) )

    module_layout_file = os.path.join( module_layouts_dir, 
                        'questions_module_layouts_{}.txt'.format(set_name) )

    question_wise_layouts = []
    with open(primary_layout_file) as f:
        for line in f.readlines():
            parse_group = line.strip()
            if parse_group != "":
                parse_strs = parse_group.split(";")
                parses = [parse_tree(p) for p in parse_strs]
                parses = [("_what", "_thing") if p == "none" else p for p in parses]
                layouts = [parse_to_layout(parse) for parse in parses]

                def flatten(lol):
                    return sum( [flatten(l) for l in lol ], [] ) if isinstance( lol, (tuple,list) ) else [lol]

                layouts = [ layout for layout in layouts if (flatten(layout).count('Describe') == 1)]
                layouts = [ layout for layout in layouts  if (len(layout) == 2 or layout[0] == 'Find') ]


                picked_layout = layouts[0] # Fix this Layout Picker to something better!!!
                question_wise_layouts.append( picked_layout )


    question_ids_file = os.path.join(question_ids_dir, 
            'questions_{}.txt'.format(set_name))

    with open(question_ids_file) as f:
        question_ids = [ line.strip() for line in f.readlines() if line.strip()]

    with open(module_layout_file, 'w') as f:
        lines = [ "\t".join([question_id, layouts]) for question_id, layouts in zip( question_ids,  map(str,question_wise_layouts)  )]        
        f.write('\n'.join(  lines  ))



