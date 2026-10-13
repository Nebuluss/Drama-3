# Utilisation rapide

Integration ComfyUI non officielle de Fish Audio Drama 3, avec les poids FP8 d'origine et une partie des experts en RAM. Testee sur RTX 5090 32 Go.

**[▶ Ecouter les exemples dans la galerie audio](https://nebuluss.github.io/Drama-3/)**

**Built with Fish Audio.** [Documentation complete](README.md) · [Exemples audio](examples/README.md) · [Workflows](workflows)

1. Installez le dossier dans `ComfyUI/custom_nodes/ComfyUI-Drama3-FP8`.
2. Lancez `install.ps1` avec le dossier du runtime et celui du checkpoint, par exemple sur `D:\Models`.
3. Copiez les trois fichiers `examples/Drama3_demo_*` dans `ComfyUI/input`.
4. Redemarrez ComfyUI et ouvrez un exemple dans les modeles de l'extension.

Pour cloner une voix, `reference_text` doit transcrire exactement `reference_audio`. Le champ `text` contient les nouvelles phrases. Pour plusieurs voix, utilisez Voice Reference et Voice Bank, puis les balises `<|speaker:0|>`, `<|speaker:1|>`, etc.

Pour un dialogue, choisissez `chunk_mode=speakers` : chaque replique utilise uniquement les references de son personnage, sans historique audio des autres voix, puis les repliques sont assemblees. Le modele est charge une seule fois. Le mode `single` peut s'arreter apres la premiere replique meme avec une fin EOS normale.

Les indications entre crochets et le champ `direction` guident le jeu. La langue depend du texte. Des exemples francais, espagnol et italien sont fournis, ainsi qu'un dialogue avec deux references et des variations de jeu.

Si le modele s'arrete apres une virgule, essayez `chunk_mode=clauses`. Chaque proposition reste envoyee au modele meme si la precedente se termine tot. Les pauses peuvent devenir plus audibles. `max_frames=0` choisit une limite automatique ; une limite atteinte peut couper la parole.

`streaming=True` active la previsualisation progressive ; cliquez Enable live listening dans le node. Le chargement reste necessaire et la lecture peut attendre entre les blocs.

La sortie `take` permet de reprendre une prise ou d'en guider une nouvelle. Save Take conserve les codes, textes et references encodees ; Load Take les recharge depuis `ComfyUI/input`. La redirection regenere une prise complete. Le workflow de remplacement d'un passage effectue un montage manuel ; l'inpainting automatique mot par mot n'est pas expose par le code public.

Voir [LICENSE](LICENSE) et [NOTICE](NOTICE) pour les conditions et l'attribution du modele.
