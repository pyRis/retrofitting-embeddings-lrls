from ppmi_generator import PPMIGenerator

# run on H100 GPU with 1Tb of memory and at least 30 CPUs if run for all languages
# run on a single GPU otherwise

def main():
    # processing ConceptNet data, building PPMI embeddings, and saving as txt
    # 'sc', 'yo', 'gn', 'qu', 'li', 'ln', 'wo', 'zu', 'rm', 'ht', 'su', 'br', 'gd', 'xh', 'mg', 'jv', 'fy', 'sa', 'my', 'ug', 'yi', 'or', 'ha', 'la', 'sd', 'so', 'ku', 
    # 'pa', 'ps', 'ga', 'am', 'km', 'uz', 'ky', 'cy', 'gu', 'eo', 'af', 'sw', 'mr', 'kn', 'ne', 'mn', 'si', 'te', 'la', 'be', 'mk', 'gl', 'hy', 'is', 'ml', 'bn', 
    # 'ur', 'kk', 'ka', 'az', 'sq', 'ta', 'et', 'lv', 'ms', 'sl', 'lt', 'he', 'sk', 'el', 'th', 'bg', 'da', 'uk', 'ro'
    
    langs = ['mt']
    for lang in langs:
        ppmi_generator = PPMIGenerator(language=lang, ppmi_dim=300)
        fetched_data = ppmi_generator.read_from_json('./exctracted_data/cn_relations_clean.json')
        print('Data ready!')
        fetched_data = ppmi_generator.process_conceptnet_language_specific(fetched_data, lang)
        print('Data processed!')
        print(f'Vocabulary for {lang} =', len(fetched_data))
        ppmi_generator.save_to_txt(fetched_data)
        print(f'Embeddings for {lang} saved!')

if __name__ == '__main__':
    main()
