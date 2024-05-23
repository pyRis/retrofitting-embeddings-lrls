# Retrofitting GloVe Embeddings for Low Resource Languages using Graph Knowledge

This project aims to enhance GloVe embeddings for low-resource languages by leveraging graph knowledge. Below is a table detailing the languages involved in the project along with their dataset sizes and classification.

## Language Dataset Details

| ISO   | Language Name     | Dataset Size () | Class |Glove Emb-s|Vocab Size|
|-------|-------------------|--------------|-------|-----------|-----------|
| ss    | Swati             | 86K          | 1     |-----------|-----------|
| sc    | Sardinian         | 143K         | 1     |-----------|-----------|
| yo    | Yoruba            | 1.1M         | 2     |-----------|-----------|
| gn    | Guarani           | 1.5M         | 1     |-----------|-----------|
| qu    | Quechua           | 1.5M         | 1     |-----------|-----------|
| ns    | Northern Sotho    | 1.8M         | 1     |-----------|-----------|
| li    | Limburgish        | 2.2M         | 1     |-----------|-----------|
| ln    | Lingala           | 2.3M         | 1     |-----------|-----------|
| wo    | Wolof             | 3.6M         | 2     |-----------|-----------|
| zu    | Zulu              | 4.3M         | 2     |-----------|-----------|
| rm    | Romansh           | 4.8M         | 1     |-----------|-----------|
| ig    | Igbo              | 6.6M         | 1     |-----------|-----------|
| lg    | Ganda             | 7.3M         | 1     |-----------|-----------|
| as    | Assamese          | 7.6M         | 1     |-----------|-----------|
| tn    | Tswana            | 8.0M         | 2     |-----------|-----------|
| ht    | Haitian           | 9.1M         | 2     |-----------|-----------|
| om    | Oromo             | 11M          | 1     |-----------|-----------|
| su    | Sundanese         | 15M          | 1     |-----------|-----------|
| bs    | Bosnian           | 18M          | 3     |-----------|-----------|
| br    | Breton            | 21M          | 1     |-----------|-----------|
| gd    | Scottish Gaelic   | 22M          | 1     |-----------|-----------|
| xh    | Xhosa             | 25M          | 2     |-----------|-----------|
| mg    | Malagasy          | 29M          | 1     |-----------|-----------|
| jv    | Javanese          | 37M          | 1     |-----------|-----------|
| fy    | Frisian           | 38M          | 0     |-----------|-----------|
| sa    | Sanskrit          | 44M          | 2     |-----------|-----------|
| my    | Burmese           | 46M          | 1     |-----------|-----------|
| ug    | Uyghur            | 46M          | 1     |-----------|-----------|
| yi    | Yiddish           | 51M          | 1     |-----------|-----------|
| or    | Oriya             | 56M          | 1     |-----------|-----------|
| ha    | Hausa             | 61M          | 2     |-----------|-----------|
| la    | Lao               | 63M          | 2     |-----------|-----------|
| sd    | Sindhi            | 67M          | 1     |-----------|-----------|
| ta_rom| Tamil Romanized   | 68M          | 3     |-----------|-----------|
| so    | Somali            | 78M          | 1     |-----------|-----------|
| te_rom| Telugu Romanized  | 79M          | 1     |-----------|-----------|
| ku    | Kurdish           | 90M          | 0     |-----------|-----------|
| pu    | Punjabi           | 90M          | 2     |-----------|-----------|
| ps    | Pashto            | 107M         | 1     |-----------|-----------|
| ga    | Irish             | 108M         | 2     |-----------|-----------|
| am    | Amharic           | 133M         | 2     |-----------|-----------|
| ur_rom| Urdu Romanized    | 141M         | 3     |-----------|-----------|
| km    | Khmer             | 153M         | 1     |-----------|-----------|
| uz    | Uzbek             | 155M         | 3     |-----------|-----------|
| bn_rom| Bengali Romanized | 164M         | 3     |-----------|-----------|
| ky    | Kyrgyz            | 173M         | 3     |-----------|-----------|
| my_zaw| Burmese (Zawgyi)  | 178M         | 1     |-----------|-----------|
| cy    | Welsh             | 179M         | 1     |-----------|-----------|
| gu    | Gujarati          | 242M         | 1     |-----------|-----------|
| eo    | Esperanto         | 250M         | 1     |-----------|-----------|
| af    | Afrikaans         | 305M         | 3     |-----------|-----------|
| sw    | Swahili           | 332M         | 2     |-----------|-----------|
| mr    | Marathi           | 334M         | 2     |-----------|-----------|
| kn    | Kannada           | 360M         | 1     |-----------|-----------|
| ne    | Nepali            | 393M         | 1     |-----------|-----------|
| mn    | Mongolian         | 397M         | 1     |-----------|-----------|
| si    | Sinhala           | 452M         | 0     |-----------|-----------|
| te    | Telugu            | 536M         | 1     |-----------|-----------|
| la    | Latin             | 609M         | 3     |-----------|-----------|
| be    | Belarussian       | 692M         | 3     |-----------|-----------|
| tl    | Tagalog           | 701M         | 3     |-----------|-----------|
| mk    | Macedonian        | 706M         | 1     |-----------|-----------|
| gl    | Galician          | 708M         | 3     |-----------|-----------|
| hy    | Armenian          | 776M         | 1     |-----------|-----------|
| is    | Icelandic         | 779M         | 2     |-----------|-----------|
| ml    | Malayalam         | 831M         | 1     |-----------|-----------|
| bn    | Bengali           | 860M         | 3     |-----------|-----------|
| ur    | Urdu              | 884M         | 3     |-----------|-----------|
| kk    | Kazakh            | 889M         | 3     |-----------|-----------|
| ka    | Georgian          | 1.1G         | 3     |-----------|-----------|
| az    | Azerbaijani       | 1.3G         | 1     |-----------|-----------|
| sq    | Albanian          | 1.3G         | 1     |-----------|-----------|
| ta    | Tamil             | 1.3G         | 3     |-----------|-----------|
| et    | Estonian          | 1.7G         | 3     |-----------|-----------|
| lv    | Latvian           | 2.1G         | 3     |-----------|-----------|
| ms    | Malay             | 2.1G         | 3     |-----------|-----------|
| sl    | Slovenian         | 2.8G         | 3     |-----------|-----------|
| lt    | Lithuanian        | 3.4G         | 3     |-----------|-----------|
| he    | Hebrew            | 6.1G         | 3     |-----------|-----------|
| sk    | Slovak            | 6.1G         | 3     |-----------|-----------|
| el    | Greek             | 7.4G         | 3     |-----------|-----------|
| th    | Thai              | 8.7G         | 3     |-----------|-----------|
| bg    | Bulgarian         | 9.3G         | 3     |-----------|-----------|
| da    | Danish            | 12G          | 3     |-----------|-----------|
| uk    | Ukrainian         | 14G          | 3     |-----------|-----------|
| ro    | Romanian          | 16G          | 3     |-----------|-----------|
| id    | Indonesian        | 36G          | 3     |-----------|-----------|

## Contributing

Contributions are welcome! Please create a pull request or open an issue for any suggestions or bug reports.

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.