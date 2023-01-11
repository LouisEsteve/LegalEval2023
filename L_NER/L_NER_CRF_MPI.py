from mpi4py import MPI
#import L_NER_CRF
from L_NER_CRF import generate_features_from_SpaCy_doc
from L_NER_CRF import features_for_CRF_from_SpaCy_doc
from L_NER_CRF import regex_annotation
from L_NER_CRF import post_processing_from_raw_offsets

comm    =   MPI.COMM_WORLD
rank    =   comm.Get_rank()
size    =   comm.Get_size()


def main() -> int:
	'''
	C-like equivalent of the main function.
	Output :
		0 -> no error
		1 -> error

	def main() -> int:
		[...]
	'''
	###
	
	########
	# Read content from files given in command line
	########
	
	corpus_objects	=	{}
	
	i	=	0
	i_limit	=	len(sys.argv)
	while i < i_limit:
		if sys.argv[i] == "--f" or sys.argv[i] == "--file":
			i	+=	1
			if sys.argv[i].endswith('.json'):
				try:
					file				=	open(sys.argv[i],"rt",encoding="UTF-8")
					corpus_content			=	json.load(file)
					corpus_objects[sys.argv[i]]	=	corpus_content
					file.close()
				except IOError or OSError as e:
					print(e)
					print(f"Could not open file {sys.argv[i]}. See error message above. Exiting.")
					exit()
			else:
				print(f"Given file ({sys.argv[i]}) is not a JSON file. Exiting.")
				exit()
		elif sys.argv[i] == "--d" or sys.argv[i] == "--directory":
			i	+=	1
			ld	=	listdir(sys.argv[i])
			for j in ld:
				if (not j.endswith('.json')) or (j.endswith('_OUTPUT.json')):
					continue
				local_path	=	f"{sys.argv[i]}{'/' if not sys.argv[i].endswith('/') else ''}{j}"
				try:
					file				=	open(local_path)
					corpus_content			=	json.load(file)
					corpus_objects[local_path]	=	corpus_content
					file.close()
				except IOError or OSError as e:
					print(e)
					print(f"Could not open file {local_path}. See error message above. Ignoring.")
		i	+=	1
	
	print(f"Found {len(corpus_objects)} elements :")
	for i in corpus_objects:
		print(i)
	
	########
	# Load configuration file that's necessary for the model
	########
	
	global config
	try:
		config_file	=	open(config_path,'rt',encoding='UTF-8')
		config		=	json.load(config_file)
		config_file.close()
		print(f'Loaded configuration from file {config_path}')
	except IOError or OSError as e:
		print(e)
		print(f'Could not start script without configuration file. Exiting.')
		return 1
	
	global CRF_model
	# prepare_model()
	
	try:
		CRF_model	=	joblib.load(config["model_path"])
		print(f'Loaded {config["model_path"]}')
	except OSError or IOError as e:
		print(e)
		print(f'Could not load model from {config["model_path"]}, see message above. Exiting.')
		return 1
	
	
	########
	# For each file given as input, process it and generate an output
	########
	
	global spacy_docs
	global doc_index
	
	nlp	=	spacy.load(config["spacy_model"])
	print(f'Loaded SpaCy with model {config["spacy_model"]}')
	
	annot_count	=	0
	file_count	=	1
	n_files		=	len(corpus_objects)
	for i in corpus_objects:
		###########################
		doc_index	=	0
		text_list	=	[]
		id_list		=	[]
	    global_text_list    =   []
	    global_id_list  =   []
        length_inventory    =   []
        standard_length =   len(global_text_list) // size
        remainder       =   len(global_text_list) % size
        for j in range(size):
            if j < remainder:
                length_inventory.append(standard_length+1)
            else:
                length_inventory.append(standard_length)
        if rank == 0:
			for j in corpus_objects[i]:
				#text_list.append(j["data"]["text"])
				#id_list.append(j["id"])
				global_text_list.append(j["data"]["text"])
				global_id_list.append(j["id"])

            # <MPI>
            #global_text_list    =   global_text_list[:]
            #global_id_list  =   global_id_list[:]
            comm.scatterv([global_text_list, length_inventory, MPI.STR], [text_list, local_length, MPI.STR], 0)
            text_list   =   global_text_list[0:local_length]
        else:
            comm.scatterv([global_text_list, length_inventory, MPI.STR], [text_list, local_length, MPI.STR], 0)
        print(f'(process {rank}) Received {len(text_list)} from scatterv : {text_list}')
        return 0
            # </MPI>
		annotation_docs	=	{}
		len_text_list	=	len(text_list)
		print('SpaCy processing:')
		for doc in nlp.pipe(text_list):
			#print(f'File {file_count}/{n_files} ({i}); text {doc_index+1}/{len_text_list} ({id_list[doc_index]})',end='\r')
			print(f'(process {rank}) File {file_count}/{n_files} ({i}); text {doc_index+1}/{len_text_list} ({id_list[doc_index]})',end='\r')
			annotation_docs[id_list[doc_index]]	=	doc
			doc_index	+=	1
		
		###########################
		
		X_features_full	=	[]
		X_offsets_full	=	[]
		
		len_i		=	len(corpus_objects[i])
		j_count		=	1
		
		########
		# Generate the features, which will then be loaded into the CRF
		########
		
		print('\nFeature generation:')
		for j in corpus_objects[i]:
			print(f'File {file_count}/{n_files} ({i}); text {j_count}/{len_i} ({j["id"]})',end='\r')
			
			j["annotations"]	=	[]
			
			text			=	j["data"]["text"]
			X_features, X_offsets	=	features_for_CRF_from_SpaCy_doc(annotation_docs[j["id"]])
			if __debug__:
				assert len(X_features) == len(X_offsets)
			
			X_features_full.append(X_features)
			X_offsets_full.append(X_offsets)
			
			j_count	+=	1	#!
		j_count	=	1
		
		########
		# Make the predictions
		########
		
		print('\nPrediction:')
		prediction_full	=	CRF_model.predict(X_features_full)
		
		########
		# Transform IOB2 annotations into raw offset annotations and place them into the JSON structure
		########
		
		occupied_offsets	=	{}
		
		for j in corpus_objects[i]:
			occupied_offsets[j["id"]]	=	[]
			print(f'File {file_count}/{n_files} ({i}); text {j_count}/{len_i} ({j["id"]})',end='\r')
			
			result_obj	=	{
				"result"	:	[]
			}
			
			text		=	j["data"]["text"]
			prediction	=	prediction_full[j_count-1]
			X_offsets	=	X_offsets_full[j_count-1]
			
			start_index	=	None
			end_index	=	None
			NE_type		=	None
			for k in range(len(prediction)):
				if prediction[k].startswith('B-') or prediction[k].startswith('S-') or prediction[k] == 'O':
					if start_index != None:
						value_obj	=	{
							"start"		:	start_index,
							"end"		:	end_index,
							"text"		:	text[start_index:end_index],
							"labels"	:	[
								NE_type
							]
						}
						result_obj["result"].append({
							"value"		:	value_obj,
							"id"		:	f"ANNOT_{annot_count}",
							"from_name"	:	"label",
							"to_name"	:	"text",
							"type"		:	"labels"
						})
						start_index	=	None
						end_index	=	None
						NE_type		=	None
						occupied_offsets[j["id"]].append((start_index,end_index))
						annot_count	+=	1
				if prediction[k].startswith('B-') or prediction[k].startswith('S-'):
					start_index	=	X_offsets[k][0]
					end_index	=	X_offsets[k][1]
					NE_type		=	prediction[k][2:]
				
				elif prediction[k].startswith('I-') or prediction[k].startswith('E-'):
					end_index	=	X_offsets[k][1]
			
			if start_index != None:
				value_obj	=	{
					"start"		:	start_index,
					"end"		:	end_index,
					"text"		:	text[start_index:end_index],
					"labels"	:	[
						NE_type
					]
				}
				result_obj["result"].append({
					"value"		:	value_obj,
					"id"		:	f"ANNOT_{annot_count}",
					"from_name"	:	"label",
					"to_name"	:	"text",
					"type"		:	"labels"
				})
				start_index	=	None
				end_index	=	None
				NE_type		=	None
				occupied_offsets[j["id"]].append((start_index,end_index))
				annot_count	+=	1

			j["annotations"].append(result_obj)
			j_count	+=	1
		
		########
		# Launch post processing from raw offsets
		########
		
		post_processing_annotations	=	post_processing_from_raw_offsets(
			raw_texts		=	text_list,
			enable_cities_query	=	"enable_cities_query" in config and config["enable_cities_query"],
			enable_DATE_regex	=	"enable_DATE_regex" in config and config["enable_DATE_regex"],
			enable_ORG_regex	=	"enable_ORG_regex" in config and config["enable_ORG_regex"]
			enable_CASE_NUMBER_regex	=	"enable_CASE_NUMBER_regex" in config and config["enable_CASE_NUMBER_regex"]
		)
		post_processing_added_annotations_count	=	0
		for j in range(len(post_processing_annotations)):
			for k in post_processing_annotations[j]:
				for m in range(len(post_processing_annotations[j][k])):
					if post_processing_annotations[j][k][m] not in occupied_offsets[id_list[j]]:
						start_index	=	post_processing_annotations[j][k][m][0]
						end_index	=	post_processing_annotations[j][k][m][1]
						value_obj	=	{
							"start"		:	start_index,
							"end"		:	end_index,
							"text"		:	text_list[j][start_index:end_index],
							"labels"	:	[
								k
							]
						}
						result_obj["result"].append({
							"value"		:	value_obj,
							"id"		:	f"ANNOT_{annot_count}",
							"from_name"	:	"label",
							"to_name"	:	"text",
							"type"		:	"labels"
						})
						annot_count	+=	1
						occupied_offsets[id_list[j]].append((start_index,end_index))
						post_processing_added_annotations_count	+=	1
		print(f'\nPost-processing added {post_processing_added_annotations_count} annotations')
		
		########
		# Write the new file
		########
		
		output_path	=	f"{i[:-5]}_OUTPUT.json"
		try:
			output_file	=	open(output_path,"wt",encoding="UTF-8")
			json.dump(corpus_objects[i],output_file,indent=8)
			print(f"\nWrote {output_path}")
		except IOError or OSError as e:
			print(e)
			print(f"\nCould not output to file {output_path}. See message above.")
			
		output_file.close()
		
		file_count	+=	1
	
	# print("Top positive:")
	# print_state_features(Counter(CRF_model.state_features_).most_common(20))
			
	return 0

if __name__ == '__main__':
    main()
