
# for name in boolean_expressions 
# for name in causal_judgement disambiguation_qa dyck_languages formal_fallacies geometric_shapes hyperbaton
# for name in logical_deduction_five_objects logical_deduction_seven_objects logical_deduction_three_objects multistep_arithmetic_two navigate
# for name in object_counting penguins_in_a_table reasoning_about_colored_objects ruin_names salient_translation_error_detection snarks
# for name in sports_understanding temporal_sequences tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects tracking_shuffled_objects_three_objects web_of_lies word_sorting
for name in date_understanding movie_recommendation
do 
    mkdir $name
    cd $name
    wget "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/$name.json"
    cd ..
done