import json
import os
import re
import warnings

import openai
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

openai.api_key = "API-KEY"

def open_ia_generation(gpt_prompt,temperature):

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": gpt_prompt}],
        temperature=temperature,
        max_tokens=2000,
        stream=False,
    )
    result=response.choices[0].message.content
    result = result.strip().rstrip()
    return result

def get_new_annotation(text):
    result={}
    correct = False
    while not correct:
        try:
            gpt_prompt = 'Esto es una conversación entre un bot y una persona. Analiza las interacciones de la persona ' \
                         'humana  en su conjunto e indica en un rango de 0.0 a 1.0: la polaridad, uso de interjecciones, ' \
                         'nivel de inseguridad en la respuesta, uso de términos de descripción de problemas de salud, ' \
                         'términos de tristeza, términos de  soledad, términos negativos, términos catastróficos, ' \
                         'repetición de conceptos. Además, di con 1 o 0 si detectas Ansiedad o Depresión. '\
                          'Siguiendo este formato JSON. No añadas ninguna explicación textual:' \
                        '{' \
                         '"polaridad":0 (0 negativa, 1 neutra, 2 positiva),' \
                        '"interjecciones":0.0,' \
                        '"inseguridad":0.0,'\
                        '"problemas_salud":0.0,' \
                         '"emociones_positivas":0.0,' \
                         '"emociones_negativas":0.0,' \
                         '"tristeza":0.0,' \
                         '"angustia":0.0,' \
                         '"soledad":0.0,' \
                        '"negativos":0.0,' \
                         '"adverbios_negativos":0.0,' \
                         '"términos_catastróficos":0.0,' \
                         '"términos_exagerados":0.0,' \
                         '"conceptos_repetidos":0.0,' \
                         '"ansiedad":0,' \
                         '"depresión":0,' \
                         '}\n "'\
                         'Conversación:' \
                         + '\n'+ text + '\n'
            result = open_ia_generation(gpt_prompt, temperature=0.0)
            result= re.sub(r"```json|```", "", result)
            result = json.loads(result, strict=False)
            print(result)
            correct = True
        except Exception as e:
            print(e)
            print("Not_found")
    return result

def get_new_annotation_from_dataset(in_path,out_path):
    dataset_chatgpt = pd.read_csv(in_path)

    list_data=[]
    for index, row in dataset_chatgpt.iterrows():

        list_data.append(get_new_annotation(row["messages"]))

    dataframe_result = pd.DataFrame(list_data)

    dataset_chatgpt=pd.concat([dataset_chatgpt, dataframe_result], axis=1)

    dataset_chatgpt.to_csv(out_path, index=False, header=True)
