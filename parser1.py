import json
from os import listdir
from os.path import exists
import re

data_path			=	'data'

encoding			=	'UTF-8'

#sentence_pattern    	 	 =   '(?s)[A-Z][^((?<![A-Z][A-Za-z]*)\\.)\\?!]*'
#sentence_regex      	 	 =   re.compile(sentence_pattern)

csv_separator       		=   '\t'
csv_replacements    		=   [
    ['\t', '\\t'],
    ['\n', '\\n'],
    ['\r', '\\r'],
    ['"', '\\"'],
    ["'", "\\'"]
]

tab				=	'\t'
tab_replacement			=	'\\t'
linefeed			=	'\n'
linefeed_replacement		=	'\\n'
carriage_return			=	'\r'
carriage_return_replacement	=	'\\r'

text_dict			=	{}

def main():
    files = listdir(data_path)
    for i in files:
        #if (not i.endswith('.json')) or (not i.startswith('RR')):
        if not i.endswith('.json'):
            continue
        local_path = f'{data_path}/{i}'
        print(local_path)
        try:
            local_file = open(local_path, 'rt', encoding='utf-8')
        except IOError or OSError as m:
            print(m)
        else:
            sum_for_average     =   0
            count_for_average   =   0
            
            json_object = json.load(local_file)
            #print(dir(json_object))

            str_for_csv = csv_separator.join(
                    [
                        'text_id',
                        # 'text',
                        'text_group',
                        'text_source',
                        'annotation_result_group',
                        'annotation_id',
                        'annotation_type',
                        'annotation_to_name',
                        'annotation_from_name',
                        'annotation_start',
                        'annotation_end',
                        'annotation_text',
                        'annotation_label'
                    ]
            )

            j_count = 1
            for j in json_object:
                #local_sentences = sentence_regex.findall(j.data.text)
                
                print(f'Parsing text {j_count}/{len(json_object)}...', end='\r')
				
                text_id                         =   j['id']             if j['id'] else hash(j['data']['text'])    			#j.id
                
                # if j['id'] not in text_dict:
                if text_id not in text_dict:
                    # text_dict[j['id']] = j['data']['text']
                    text_dict[text_id] = j['data']['text']
                
                sum_for_average                 +=  len(j['data']['text'])
                count_for_average               +=  1
                

                # text_id                         =   j['id']             if len(j['id']) > 0 else hash(j['data']['text'])    #j.id
                # text                          =   j['data']['text']                                                       #j.data.text
                text_group                      =   j['meta']['group']  if 'group' in j['meta'].keys() else '_'             #j.meta.group
                text_source                     =   j['meta']['source'] if 'source' in j['meta'].keys() else '_'            #j.meta.group

                for q in range(len(j['annotations'])):
                    annotation_result_group     =    q
                    
                    #for k in j['annotations'][0]['result']:
                    for k in j['annotations'][q]['result']:
                        annotation_id           =   k['id']                                                                 #k.id
                        annotation_type         =   k['type']           if 'type' in k.keys() else '_'                      #k.type
                        annotation_to_name      =   k['to_name']        if 'to_name' in k.keys() else '_'                   #k.to_name
                        annotation_from_name    =   k['from_name']      if 'from_name' in k.keys() else '_'                 #k.from_name
                        annotation_start        =   k['value']['start']                                                     #k.value.start
                        annotation_end          =   k['value']['end']                                                       #k.value.end
                        annotation_text         =   k['value']['text']                                                      #k.value.text

                        for m in k['value']['labels']:
                            annotation_label    =   m
                            
                            cell_values         =   [
                                str(text_id),
                                # str(text),
                                str(text_group),
                                str(text_source),
                                str(annotation_result_group),
                                str(annotation_id),
                                str(annotation_type),
                                str(annotation_to_name),
                                str(annotation_from_name),
                                str(annotation_start),
                                str(annotation_end),
                                str(annotation_text),
                                str(annotation_label)
                            ]
                            
                            
                            #for n in cell_values:
                            #    for p in csv_replacements:
                            #        n           =   n.replace(p[0], p[1])

                            for n in range(len(cell_values)):
                                for p in csv_replacements:
                                    cell_values[n]  =   cell_values[n].replace(p[0], p[1])

                            local_line_for_csv  =   csv_separator.join(cell_values)

                            str_for_csv         =   f'{str_for_csv}\n{local_line_for_csv}'
                j_count += 1
            local_file.close()
            print('')
            
            output_count = 0
            output_path = f'{local_path[:-5]}_restructured{str(output_count)}.csv'
            while(exists(output_path)):
                output_count += 1
                output_path = f'{local_path[:-5]}_restructured{str(output_count)}.csv'
            f = open(output_path, 'wt', encoding='utf-8')
            f.write(str_for_csv)
            f.close()
            print(f'Wrote {output_path} (character count: {len(str_for_csv)})')
            print(f'Average length: {sum_for_average/count_for_average}\n{"-"*32}')
            #break
    text_base_path  =   f'{data_path}/text_base.csv'
    f               =   open(text_base_path,'wt',encoding=encoding)
    # s               =   '\n'.join(['\t'.join([i,text_dict[i].replace('\t','\\t').replace('\n','\\n') for i in text_dict])])
    # s               =   'id\ttext\n' + '\n'.join([f"{i}\t{text_dict[i].replace(tab,tab_replacement).replace(linefeed,linefeed_replacement).replace(carriage_return,carriage_return_replacement)}" for i in text_dict])
    for i in text_dict:
        for j in csv_replacements:
            text_dict[i]	=	text_dict[i].replace(j[0],j[1])
    s               =   'id\ttext\n' + '\n'.join([f"{i}\t{text_dict[i]}" for i in text_dict])
    f.write(s)
    print(f'Wrote {text_base_path}')
    f.close()


if __name__ == '__main__':
    main()

