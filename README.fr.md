# Utilisation rapide

Integration ComfyUI non officielle de Fish Audio Drama 3, avec les poids FP8 d'origine et une partie des experts en RAM. Testee sur RTX 5090 32 Go.

**[▶ Ecouter les exemples dans la galerie audio](https://nebuluss.github.io/Drama-3/)**

**Built with Fish Audio.** [Documentation complete](README.md) · [Exemples audio](examples/README.md) · [Workflows](workflows)

1. Installez le dossier dans `ComfyUI/custom_nodes/ComfyUI-Drama3-FP8`.
2. Lancez `install.ps1` avec le dossier du runtime et celui du checkpoint, par exemple sur `D:\Models`.
3. Copiez les trois fichiers `examples/Drama3_demo_*` dans `ComfyUI/input`.
4. Redemarrez ComfyUI et ouvrez un exemple dans les modeles de l'extension.

Pour cloner une voix, `reference_text` doit transcrire exactement `reference_audio`. Le champ `text` contient les nouvelles phrases. Pour plusieurs voix, utilisez Voice Reference et Voice Bank, puis les balises `<|speaker:0|>`, `<|speaker:1|>`, etc.

Le [workflow de clonage avec Whisper](workflows/Drama3-Whisper-Voice-Cloning.json) transcrit la reference automatiquement avec **Load Whisper (mtb) → Audio To Text (mtb)**, du pack [comfy_mtb](https://github.com/melMass/comfy_mtb). Il attend un modele Whisper complet dans `ComfyUI/models/whisper/whisper-large`, utilise `large`, `fr`, sans timestamps et sans telechargement. Le meme audio alimente Whisper et `reference_audio` ; la sortie `STRING` va a `reference_text`. Pour creer cette entree manuellement : clic droit sur Drama 3 → Convert Widget to Input → reference_text. Le Whisper Model Loader de WanVideo ne produit pas de transcription.

Apres une mise a jour des nodes, arretez puis redemarrez **le processus ComfyUI**, puis rechargez le workflow. Un rafraichissement du navigateur ne suffit pas. Une erreur qui refuse `max_frames=0` ou cherche le runtime dans le dossier ComfyUI indique qu'une ancienne version du node est encore chargee. Le runtime est configure dans `runtime.json`, ou explicitement dans `runtime_folder`.

Pour un dialogue, choisissez `chunk_mode=speakers` : chaque replique utilise uniquement les references de son personnage, sans historique audio des autres voix, puis les repliques sont assemblees. Le modele est charge une seule fois. Le mode `single` peut s'arreter apres la premiere replique meme avec une fin EOS normale.

Les indications entre crochets et le champ `direction` guident le jeu. La langue depend du texte. Des exemples francais, espagnol et italien sont fournis, ainsi qu'un dialogue avec deux references et des variations de jeu.

Si le modele s'arrete apres une virgule, essayez `chunk_mode=clauses`. Chaque proposition reste envoyee au modele meme si la precedente se termine tot. Les pauses peuvent devenir plus audibles. `max_frames=0` choisit une limite automatique ; une limite atteinte peut couper la parole.

Pour un monologue plus fluide, commencez par `chunk_mode=auto` et `chunk_bytes=300`. Le mode `clauses` coupe aussi aux points, deux-points, points d'interrogation/exclamation et retours a la ligne : gardez-le pour depanner les arrets prematures. Le workflow Whisper utilise `auto` par defaut.

`streaming=True` active la previsualisation progressive ; cliquez Enable live listening dans le node. Le chargement reste necessaire et la lecture peut attendre entre les blocs.

Activez `streaming=True` **avant Run**. Le bouton Enable live listening autorise seulement la lecture ; il ne change pas une generation deja lancee. Avec `streaming=False`, le lecteur live ne recoit aucun bloc et le fichier final arrive a la fin.

La sortie `take` permet de reprendre une prise ou d'en guider une nouvelle. Save Take conserve les codes, textes et references encodees ; Load Take les recharge depuis `ComfyUI/input`. La redirection regenere une prise complete. Le workflow de remplacement d'un passage effectue un montage manuel ; l'inpainting automatique mot par mot n'est pas expose par le code public.

Voir [LICENSE](LICENSE) et [NOTICE](NOTICE) pour les conditions et l'attribution du modele.
