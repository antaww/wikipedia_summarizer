import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string

# 1. Charger et prétraiter le texte
texte = """La chimie est une science de la nature qui étudie la matière et ses transformations, et plus précisément les atomes, les molécules, les réactions chimiques et les forces qui favorisent les réactions chimiques.

Présentation générale
La chimie porte sur les sujets suivants : 

les entités chimiques d'un élément, c'est-à-dire les atomes et les ions monoatomiques. Elle étudie également leurs associations par liaisons chimiques qui engendrent notamment des composés moléculaires stables ou des intermédiaires plus ou moins instables. Ces entités de matière peuvent être caractérisées par une identité reliée à des caractéristiques quantiques et des propriétés précises ;
les processus qui changent ou modifient l'identité de ces particules ou molécules de matière, dénommés réaction chimique, transformation, interaction, etc. ;
les mécanismes réactionnels intervenant dans les processus chimiques ou les équilibres physiques entre deux formes, qui permettent d'interpréter des observations et d'envisager de nouvelles réactions ;
les phénomènes fondamentaux observables en rapport avec les forces de la nature qui jouent un rôle chimique, favorisant les réactions ou synthèses, addition, combinaison ou décomposition, séparation de phases ou extraction. L'analyse permet de découvrir les compositions, le marquage sélectif ouvre la voie à un schéma réactionnel cohérent dans des mélanges complexes.
La taille des entités chimiques varie de simples atomes ou molécules nanométriques aux édifices moléculaires de plusieurs dizaines de milliers d'atomes dans les macromolécules, l'ADN ou protéine de la matière vivante (infra)micrométrique, jusqu'à des dimensions parfois macroscopiques des cristaux. En incluant l'électron libre (qui intervient dans les réactions radicalaires), les dimensions de principaux domaines d'application se situent dans son ensemble entre le femtomètre (10−15 m) et le micromètre (10−6 m).
L'étude du monde à l'échelle moléculaire soumise paradoxalement à des lois singulières, comme le prouvent les récents développements nanotechnologiques, permet de mieux comprendre les détails de notre monde macroscopique. La chimie est qualifiée de « science centrale » en raison des relations étroites qu'elle possède avec la biologie et la physique. Et elle a des relations avec les champs d'applications variés, tels que  la médecine, la pharmacie, l'informatique et la science des matériaux, sans oublier des domaines appliqués tels que le génie des procédés et toutes les activités de formulation.
La physique, et surtout son instrumentation, est devenue hégémonique après 1950 dans le champ de la science de la nature. Les avancées en physique ont surtout refondé en partie la chimie physique et la chimie inorganique. La chimie organique, par l'intermédiaire de la biochimie, a partagé des recherches valorisant la biologie. Mais la chimie n'en garde pas moins une place incontournable et légitime dans le champ des sciences de la nature : elle conduit à de nouveaux produits, de nouveaux composés, découvre ou invente des structures moléculaires simples ou complexes qui bénéficient de façon extraordinaire à la recherche physique ou biologique. Enfin l'héritage cohérent que les chimistes défenseurs marginaux des structures atomiques ont légué aux acteurs de la révolution des conceptions physiciennes au début du XXe siècle ne doit pas être sous-estimé.

Étymologie
Trois étymologies sont fréquemment citées[Par qui ?], mais ces hypothèses peuvent être reliées :

l'une égyptienne, kemi viendrait de l'ancien égyptien Khemet, la terre. Il se retrouve aussi dans le copte chame « noire » puisque dans la vallée du Nil, la terre est noire. L'art de la kemi, par exemple les poisons minéraux, a pu influencer la magie noire ;
la racine grecque se lie à χυμεία, khumeia, « mélange de liquides » (χυμός, khumos, « suc, jus ») ;
enfin, le mot « chimie » proviendrait de l'arabe al kemi, الكيمياء (littéralement la kemia, la « chimie »), venant du grec χεμεία / khemeía[Quoi ?]<!- ce mot n'existe ni dans le Bailly ni dans le Liddell-Scott...  -->, « magie noire »[réf. nécessaire], mot lui-même venant de l'égyptien ancien kem qui désigne la couleur noire.

Histoire
Antiquité et Moyen-Âge
L'art d'employer ou de trier, préparer, purifier, transformer les substances séchées mises sous forme de poudres, qu'elles proviennent du désert ou de vallées sèches, a donné naissance à des codifications savantes, d'abord essentiellement minérales. Mais les plantes éphémères et les arbres pérennes du désert, et leurs extraits gommeux ou liquides nécessaires aux onguents, ont été très vite assimilés à celles-ci, par reconnaissance de l'influence des terres et des roches.
Outre la connaissance du cycle de l'eau et des transports sédimentaires, la maîtrise progressive des métaux et des terres, les Égyptiens de l'Antiquité connaissent beaucoup de choses. Parmi elles, le plâtre, le verre, la potasse, les vernis, le papier (papyrus durci à l'amidon), l'encens, une vaste gamme de couleurs minérales ou pigments, de remèdes et de produits cosmétiques, etc. Plus encore que les huiles à onction ou les bains d'eaux ou de boues relaxants ou guérisseurs, la chimie se présente comme un savoir sacré qui permet la survie. Par exemple par l'art sophistiqué d'embaumer ou par le placement des corps des plus humbles dans un endroit sec.
L'art de la terre égyptien a été enseigné en préservant une conception unitaire. Les temples et les administrations religieuses ont préservé et parfois figé le meilleur des savoirs. Le pouvoir politique souverain s'est appuyé sur les mesures physiques, arpentage et hauteur hydraulique des crues, peut-être sur la densité du limon en suspension, pour déterminer l'impôt et sur les matériaux permettant les déplacements ou la mobilité des armées. Le vitalisme ou les cultes agraires et animaux, domaines appliqués de la kemia, ont été préservés dans des temples, à l'instar d'Amon, conservatoire des fumures azotées et de la chimie ammoniacale antique.

Les savants musulmans supposaient que tous les métaux provenaient de la même espèce. Ils croyaient à la possibilité de la transmutation et cherchèrent en vain dans cette perspective l'obtention de « l'al-iksir » qui prolongerait la vie.

« Dans le même temps, guidés par des préoccupations plus pratiques, ils se livraient dans leurs laboratoires à des expérimentations systématiques des corps. Disposant de tableaux indiquant les poids spécifiques, ils pouvaient en les pesant, les distinguer, les reconnaître par des analyses sommaires et, quelquefois même les reconstituer par synthèse. [...] Ils trouvèrent des teintures pour colorer les tissus, les mosaïques et les peintures, si parfaites qu'elles ont gardé leur fraîcheur millénaire. »
« Les Arabes allaient faire connaître au monde l'usage des parfums, en apprenant à extraire les parfums des fleurs. À Chapur, on distillait toutes les essences selon les techniques zoroastriennes : narcisse, lilas, violette, jasmin… Gur était réputé pour ses eaux parfumées et fabriquait des eaux de fleur d'oranger et de rose à base de rose d'Ispahan. Samarkand était célèbre par son parfum de basilic, Sikr par son ambre. Le musc du Tibet, le Nénuphar d'Albanie, la Rose de Perse demeurent des parfums aussi prestigieux que légendaires. »
« En mélangeant la soude (Al-qali) avec le suif ou l'huile, les Arabes fabriquèrent les premiers savons et créèrent une des plus magnifiques industries de Bagdad, qui devait s'étendre rapidement sur l'Égypte, la Syrie, La Tunisie et l'Espagne musulmane. L'islam avait fait si bien que le goût du bien-être gagna toutes les classes de la société et que la production ne suffit plus à la consommation. Le besoin d'inventer l'industrie des succédanés ou ersatz se fit sentir à ce moment-là » »
Les repères de pensée taxonomique sont profondément influencés par les civilisations grecques puis hellénistiques, férues de théorisations, qui ont lentement esquissé de façon sommaire ce qui encadre aux yeux profanes la chimie, la physique et la biologie. Elles ont laissé les techniques vulgaires au monde du travail et de l'esclave. L'émergence de spiritualités populaires, annexant l'utile à des cultes hermétiques, a promu et malaxé ses bribes de savoirs dispersés. Incontestablement, les premiers textes datés tardivement du Ier siècle et IIe siècle après Jésus-Christ comportent à l'exemple de l'alchimie médiévale la plus ésotérique, une partie mystique et une partie opératoire. La religiosité hellénistique a ainsi légué aussi bien le bain-marie, de Marie la Juive que l'abscons patronage d'Hermès Trismégiste, divinité qui prétendait expliquer à la fois le mouvement et la stabilité de toute chose humaine, terrestre ou céleste.

De l'alchimie pré-scientifique à la chimie scientifique
Au cours des siècles, ce savoir empirique oscille entre art sacré et pratique profane. Il s'est préservé comme l'atteste le vocable chimia des scolastiques en 1356, mais savoir et savoir-faire sont souvent segmentés à l'extrême. Parfois, il est amélioré dans le monde paysan, artisan ou minier avant de devenir une science expérimentale, la chimie, au cours des troisième et quatrième décennies du XVIIe siècle.
Au même titre que la physique, l'essor de la pensée et de la modélisation mécanistes, font naître la chimie sous forme de science expérimentale et descriptive. La chimie reste essentiellement qualitative et bute sur le retour incessant des croyances écartées.
Les alchimistes ont subsisté jusqu'en 1850. Ils poursuivaient la quête de la pierre philosophale et continuant l'alchimie sous une forme ésotérique.
La rupture entre la chimie et l'alchimie apparaît pourtant clairement en 1722, quand Étienne Geoffroy l'Aîné, médecin et naturaliste français, affirme l'impossibilité de la transmutation. La chimie expérimentale et l'alchimie diffèrent déjà radicalement. La première est de nature scientifique alors que la seconde représente un ensemble de croyances non-scientifiques.

La chimie a connu une avancée énorme """
# S’assurer d’avoir téléchargé les ressources NLTK :
nltk.download('punkt')
nltk.download('stopwords')

# Conversion en minuscules
texte_clean = texte.lower()
# Suppression de la ponctuation
texte_clean = texte_clean.translate(str.maketrans('', '', string.punctuation))

# 2. Découpage en phrases
phrases = sent_tokenize(texte)

# 3. Tokenisation et fréquences
mots = word_tokenize(texte_clean)
stop_fr = set(stopwords.words('french'))
mots_filtrés = [m for m in mots if m not in stop_fr and m.isalpha()]

freq_dist = nltk.FreqDist(mots_filtrés)

# 4. Score des phrases
scores = {}
for sentence in phrases:
    # Tokeniser la phrase (en minuscules, sans ponctuation)
    tokens = word_tokenize(sentence.lower().translate(str.maketrans('', '', string.punctuation)))
    score = sum(freq_dist[word] for word in tokens if word in freq_dist)
    # Optionnel : normaliser par longueur
    if len(tokens) > 0:
        score /= len(tokens)
    scores[sentence] = score

# 5. Sélection des N phrases les mieux scorées
N = 3  # par exemple
meilleures = sorted(scores, key=scores.get, reverse=True)[:N]

# Reconstruction du résumé
résumé = ' '.join([phrase for phrase in phrases if phrase in meilleures])
print("Résumé :\n", résumé)