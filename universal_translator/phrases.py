from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pycountry
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

PHRASE_DATABASE_JSON = r"""
{
  "phrases": [
    {
      "latin": "salve",
      "category": "word",
      "translations": {
        "eng_Latn": "hello",
        "fra_Latn": "bonjour",
        "spa_Latn": "hola",
        "deu_Latn": "hallo",
        "ita_Latn": "ciao",
        "por_Latn": "ola"
      }
    },
    {
      "latin": "vale",
      "category": "word",
      "translations": {
        "eng_Latn": "farewell",
        "fra_Latn": "adieu",
        "spa_Latn": "adios",
        "deu_Latn": "leb wohl",
        "ita_Latn": "addio",
        "por_Latn": "adeus"
      }
    },
    {
      "latin": "amicus",
      "category": "word",
      "translations": {
        "eng_Latn": "friend",
        "fra_Latn": "ami",
        "spa_Latn": "amigo",
        "deu_Latn": "freund",
        "ita_Latn": "amico",
        "por_Latn": "amigo"
      }
    },
    {
      "latin": "aqua",
      "category": "word",
      "translations": {
        "eng_Latn": "water",
        "fra_Latn": "eau",
        "spa_Latn": "agua",
        "deu_Latn": "wasser",
        "ita_Latn": "acqua",
        "por_Latn": "agua"
      }
    },
    {
      "latin": "terra",
      "category": "word",
      "translations": {
        "eng_Latn": "earth",
        "fra_Latn": "terre",
        "spa_Latn": "tierra",
        "deu_Latn": "erde",
        "ita_Latn": "terra",
        "por_Latn": "terra"
      }
    },
    {
      "latin": "mare",
      "category": "word",
      "translations": {
        "eng_Latn": "sea",
        "fra_Latn": "mer",
        "spa_Latn": "mar",
        "deu_Latn": "meer",
        "ita_Latn": "mare",
        "por_Latn": "mar"
      }
    },
    {
      "latin": "ignis",
      "category": "word",
      "translations": {
        "eng_Latn": "fire",
        "fra_Latn": "feu",
        "spa_Latn": "fuego",
        "deu_Latn": "feuer",
        "ita_Latn": "fuoco",
        "por_Latn": "fogo"
      }
    },
    {
      "latin": "lux",
      "category": "word",
      "translations": {
        "eng_Latn": "light",
        "fra_Latn": "lumiere",
        "spa_Latn": "luz",
        "deu_Latn": "licht",
        "ita_Latn": "luce",
        "por_Latn": "luz"
      }
    },
    {
      "latin": "nox",
      "category": "word",
      "translations": {
        "eng_Latn": "night",
        "fra_Latn": "nuit",
        "spa_Latn": "noche",
        "deu_Latn": "nacht",
        "ita_Latn": "notte",
        "por_Latn": "noite"
      }
    },
    {
      "latin": "sol",
      "category": "word",
      "translations": {
        "eng_Latn": "sun",
        "fra_Latn": "soleil",
        "spa_Latn": "sol",
        "deu_Latn": "sonne",
        "ita_Latn": "sole",
        "por_Latn": "sol"
      }
    },
    {
      "latin": "luna",
      "category": "word",
      "translations": {
        "eng_Latn": "moon",
        "fra_Latn": "lune",
        "spa_Latn": "luna",
        "deu_Latn": "mond",
        "ita_Latn": "luna",
        "por_Latn": "lua"
      }
    },
    {
      "latin": "vita",
      "category": "word",
      "translations": {
        "eng_Latn": "life",
        "fra_Latn": "vie",
        "spa_Latn": "vida",
        "deu_Latn": "leben",
        "ita_Latn": "vita",
        "por_Latn": "vida"
      }
    },
    {
      "latin": "mors",
      "category": "word",
      "translations": {
        "eng_Latn": "death",
        "fra_Latn": "mort",
        "spa_Latn": "muerte",
        "deu_Latn": "tod",
        "ita_Latn": "morte",
        "por_Latn": "morte"
      }
    },
    {
      "latin": "amor",
      "category": "word",
      "translations": {
        "eng_Latn": "love",
        "fra_Latn": "amour",
        "spa_Latn": "amor",
        "deu_Latn": "liebe",
        "ita_Latn": "amore",
        "por_Latn": "amor"
      }
    },
    {
      "latin": "pax",
      "category": "word",
      "translations": {
        "eng_Latn": "peace",
        "fra_Latn": "paix",
        "spa_Latn": "paz",
        "deu_Latn": "frieden",
        "ita_Latn": "pace",
        "por_Latn": "paz"
      }
    },
    {
      "latin": "bellum",
      "category": "word",
      "translations": {
        "eng_Latn": "war",
        "fra_Latn": "guerre",
        "spa_Latn": "guerra",
        "deu_Latn": "krieg",
        "ita_Latn": "guerra",
        "por_Latn": "guerra"
      }
    },
    {
      "latin": "veritas",
      "category": "word",
      "translations": {
        "eng_Latn": "truth",
        "fra_Latn": "verite",
        "spa_Latn": "verdad",
        "deu_Latn": "wahrheit",
        "ita_Latn": "verita",
        "por_Latn": "verdade"
      }
    },
    {
      "latin": "sapientia",
      "category": "word",
      "translations": {
        "eng_Latn": "wisdom",
        "fra_Latn": "sagesse",
        "spa_Latn": "sabiduria",
        "deu_Latn": "weisheit",
        "ita_Latn": "saggezza",
        "por_Latn": "sabedoria"
      }
    },
    {
      "latin": "fortuna",
      "category": "word",
      "translations": {
        "eng_Latn": "fortune",
        "fra_Latn": "fortune",
        "spa_Latn": "fortuna",
        "deu_Latn": "gluck",
        "ita_Latn": "fortuna",
        "por_Latn": "fortuna"
      }
    },
    {
      "latin": "virtus",
      "category": "word",
      "translations": {
        "eng_Latn": "virtue",
        "fra_Latn": "vertu",
        "spa_Latn": "virtud",
        "deu_Latn": "tugend",
        "ita_Latn": "virtu",
        "por_Latn": "virtude"
      }
    },
    {
      "latin": "tempus",
      "category": "word",
      "translations": {
        "eng_Latn": "time",
        "fra_Latn": "temps",
        "spa_Latn": "tiempo",
        "deu_Latn": "zeit",
        "ita_Latn": "tempo",
        "por_Latn": "tempo"
      }
    },
    {
      "latin": "dies",
      "category": "word",
      "translations": {
        "eng_Latn": "day",
        "fra_Latn": "jour",
        "spa_Latn": "dia",
        "deu_Latn": "tag",
        "ita_Latn": "giorno",
        "por_Latn": "dia"
      }
    },
    {
      "latin": "via",
      "category": "word",
      "translations": {
        "eng_Latn": "road",
        "fra_Latn": "route",
        "spa_Latn": "camino",
        "deu_Latn": "weg",
        "ita_Latn": "via",
        "por_Latn": "caminho"
      }
    },
    {
      "latin": "domus",
      "category": "word",
      "translations": {
        "eng_Latn": "home",
        "fra_Latn": "maison",
        "spa_Latn": "casa",
        "deu_Latn": "haus",
        "ita_Latn": "casa",
        "por_Latn": "casa"
      }
    },
    {
      "latin": "rex",
      "category": "word",
      "translations": {
        "eng_Latn": "king",
        "fra_Latn": "roi",
        "spa_Latn": "rey",
        "deu_Latn": "konig",
        "ita_Latn": "re",
        "por_Latn": "rei"
      }
    },
    {
      "latin": "regina",
      "category": "word",
      "translations": {
        "eng_Latn": "queen",
        "fra_Latn": "reine",
        "spa_Latn": "reina",
        "deu_Latn": "konigin",
        "ita_Latn": "regina",
        "por_Latn": "rainha"
      }
    },
    {
      "latin": "canis",
      "category": "word",
      "translations": {
        "eng_Latn": "dog",
        "fra_Latn": "chien",
        "spa_Latn": "perro",
        "deu_Latn": "hund",
        "ita_Latn": "cane",
        "por_Latn": "cao"
      }
    },
    {
      "latin": "feles",
      "category": "word",
      "translations": {
        "eng_Latn": "cat",
        "fra_Latn": "chat",
        "spa_Latn": "gato",
        "deu_Latn": "katze",
        "ita_Latn": "gatto",
        "por_Latn": "gato"
      }
    },
    {
      "latin": "equus",
      "category": "word",
      "translations": {
        "eng_Latn": "horse",
        "fra_Latn": "cheval",
        "spa_Latn": "caballo",
        "deu_Latn": "pferd",
        "ita_Latn": "cavallo",
        "por_Latn": "cavalo"
      }
    },
    {
      "latin": "avis",
      "category": "word",
      "translations": {
        "eng_Latn": "bird",
        "fra_Latn": "oiseau",
        "spa_Latn": "pajaro",
        "deu_Latn": "vogel",
        "ita_Latn": "uccello",
        "por_Latn": "passaro"
      }
    },
    {
      "latin": "carpe diem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "seize the day",
        "fra_Latn": "profite du jour",
        "spa_Latn": "aprovecha el dia",
        "deu_Latn": "nutze den tag",
        "ita_Latn": "cogli l attimo",
        "por_Latn": "aproveita o dia"
      }
    },
    {
      "latin": "alea iacta est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the die is cast",
        "fra_Latn": "le sort en est jete",
        "spa_Latn": "la suerte esta echada",
        "deu_Latn": "der wurfel ist gefallen",
        "ita_Latn": "il dado e tratto",
        "por_Latn": "a sorte esta lancada"
      }
    },
    {
      "latin": "veni vidi vici",
      "category": "phrase",
      "translations": {
        "eng_Latn": "i came i saw i conquered",
        "fra_Latn": "je suis venu j ai vu j ai vaincu",
        "spa_Latn": "vine vi vi venci",
        "deu_Latn": "ich kam ich sah ich siegte",
        "ita_Latn": "venni vidi vinsi",
        "por_Latn": "vim vi venci"
      }
    },
    {
      "latin": "et tu brute",
      "category": "phrase",
      "translations": {
        "eng_Latn": "you too brutus",
        "fra_Latn": "toi aussi brutus",
        "spa_Latn": "tu tambien bruto",
        "deu_Latn": "auch du brutus",
        "ita_Latn": "anche tu bruto",
        "por_Latn": "tu tambem bruto"
      }
    },
    {
      "latin": "in vino veritas",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in wine there is truth",
        "fra_Latn": "dans le vin la verite",
        "spa_Latn": "en el vino esta la verdad",
        "deu_Latn": "im wein liegt die wahrheit",
        "ita_Latn": "nel vino c e la verita",
        "por_Latn": "no vinho esta a verdade"
      }
    },
    {
      "latin": "per aspera ad astra",
      "category": "phrase",
      "translations": {
        "eng_Latn": "through hardships to the stars",
        "fra_Latn": "par les difficultes jusqu aux etoiles",
        "spa_Latn": "a traves de las dificultades hasta las estrellas",
        "deu_Latn": "durch schwierigkeiten zu den sternen",
        "ita_Latn": "attraverso le difficolta fino alle stelle",
        "por_Latn": "pelas dificuldades ate as estrelas"
      }
    },
    {
      "latin": "amor vincit omnia",
      "category": "phrase",
      "translations": {
        "eng_Latn": "love conquers all",
        "fra_Latn": "l amour triomphe de tout",
        "spa_Latn": "el amor todo lo vence",
        "deu_Latn": "liebe besiegt alles",
        "ita_Latn": "l amore vince tutto",
        "por_Latn": "o amor vence tudo"
      }
    },
    {
      "latin": "memento mori",
      "category": "phrase",
      "translations": {
        "eng_Latn": "remember that you must die",
        "fra_Latn": "souviens toi que tu dois mourir",
        "spa_Latn": "recuerda que debes morir",
        "deu_Latn": "gedenke dass du sterben musst",
        "ita_Latn": "ricorda che devi morire",
        "por_Latn": "lembra te de que deves morrer"
      }
    },
    {
      "latin": "dum spiro spero",
      "category": "phrase",
      "translations": {
        "eng_Latn": "while i breathe i hope",
        "fra_Latn": "tant que je respire j espere",
        "spa_Latn": "mientras respiro tengo esperanza",
        "deu_Latn": "solange ich atme hoffe ich",
        "ita_Latn": "finche respiro spero",
        "por_Latn": "enquanto respiro espero"
      }
    },
    {
      "latin": "audentes fortuna iuvat",
      "category": "phrase",
      "translations": {
        "eng_Latn": "fortune favors the bold",
        "fra_Latn": "la fortune favorise les audacieux",
        "spa_Latn": "la fortuna favorece a los audaces",
        "deu_Latn": "das gluck begunstigt die mutigen",
        "ita_Latn": "la fortuna aiuta gli audaci",
        "por_Latn": "a fortuna favorece os audazes"
      }
    },
    {
      "latin": "sic transit gloria mundi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "thus passes the glory of the world",
        "fra_Latn": "ainsi passe la gloire du monde",
        "spa_Latn": "asi pasa la gloria del mundo",
        "deu_Latn": "so vergeht der ruhm der welt",
        "ita_Latn": "cosi passa la gloria del mondo",
        "por_Latn": "assim passa a gloria do mundo"
      }
    },
    {
      "latin": "errare humanum est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "to err is human",
        "fra_Latn": "l erreur est humaine",
        "spa_Latn": "errar es humano",
        "deu_Latn": "irren ist menschlich",
        "ita_Latn": "errare e umano",
        "por_Latn": "errar e humano"
      }
    },
    {
      "latin": "festina lente",
      "category": "phrase",
      "translations": {
        "eng_Latn": "make haste slowly",
        "fra_Latn": "hate toi lentement",
        "spa_Latn": "apresurate despacio",
        "deu_Latn": "eile mit weile",
        "ita_Latn": "affrettati lentamente",
        "por_Latn": "apressa te devagar"
      }
    },
    {
      "latin": "labor omnia vincit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "work conquers all",
        "fra_Latn": "le travail triomphe de tout",
        "spa_Latn": "el trabajo todo lo vence",
        "deu_Latn": "arbeit uberwindet alles",
        "ita_Latn": "il lavoro vince tutto",
        "por_Latn": "o trabalho vence tudo"
      }
    },
    {
      "latin": "scientia potentia est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "knowledge is power",
        "fra_Latn": "le savoir est le pouvoir",
        "spa_Latn": "el conocimiento es poder",
        "deu_Latn": "wissen ist macht",
        "ita_Latn": "la conoscenza e potere",
        "por_Latn": "conhecimento e poder"
      }
    },
    {
      "latin": "ubi concordia ibi victoria",
      "category": "phrase",
      "translations": {
        "eng_Latn": "where there is unity there is victory",
        "fra_Latn": "la ou il y a l unite il y a la victoire",
        "spa_Latn": "donde hay unidad hay victoria",
        "deu_Latn": "wo einigkeit ist ist sieg",
        "ita_Latn": "dove c e unita c e vittoria",
        "por_Latn": "onde ha unidade ha vitoria"
      }
    },
    {
      "latin": "omnia vincit amor",
      "category": "phrase",
      "translations": {
        "eng_Latn": "love conquers all things",
        "fra_Latn": "l amour triomphe de toutes choses",
        "spa_Latn": "el amor vence todas las cosas",
        "deu_Latn": "liebe besiegt alle dinge",
        "ita_Latn": "l amore vince ogni cosa",
        "por_Latn": "o amor vence todas as coisas"
      }
    },
    {
      "latin": "ars longa vita brevis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "art is long life is short",
        "fra_Latn": "l art est long la vie est breve",
        "spa_Latn": "el arte es largo la vida es breve",
        "deu_Latn": "die kunst ist lang das leben ist kurz",
        "ita_Latn": "l arte e lunga la vita e breve",
        "por_Latn": "a arte e longa a vida e breve"
      }
    },
    {
      "latin": "historia magistra vitae est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "history is the teacher of life",
        "fra_Latn": "l histoire est la maitresse de la vie",
        "spa_Latn": "la historia es la maestra de la vida",
        "deu_Latn": "die geschichte ist die lehrerin des lebens",
        "ita_Latn": "la storia e maestra di vita",
        "por_Latn": "a historia e a mestra da vida"
      }
    },
    {
      "latin": "ab imo pectore",
      "category": "phrase",
      "translations": {
        "eng_Latn": "from the bottom of my heart",
        "fra_Latn": "du fond du coeur",
        "spa_Latn": "desde el fondo del corazon",
        "deu_Latn": "von ganzem herzen",
        "ita_Latn": "dal profondo del cuore",
        "por_Latn": "do fundo do coracao"
      }
    },
    {
      "latin": "ab initio",
      "category": "phrase",
      "translations": {
        "eng_Latn": "from the beginning",
        "fra_Latn": "depuis le debut",
        "spa_Latn": "desde el principio",
        "deu_Latn": "von anfang an",
        "ita_Latn": "dall inizio",
        "por_Latn": "desde o inicio"
      }
    },
    {
      "latin": "absit omen",
      "category": "phrase",
      "translations": {
        "eng_Latn": "may it not be an omen",
        "fra_Latn": "que cela ne soit pas un presage",
        "spa_Latn": "que no sea un presagio",
        "deu_Latn": "moge es kein omen sein",
        "ita_Latn": "che non sia un presagio",
        "por_Latn": "que nao seja um pressagio"
      }
    },
    {
      "latin": "acta non verba",
      "category": "phrase",
      "translations": {
        "eng_Latn": "deeds not words",
        "fra_Latn": "des actes pas des paroles",
        "spa_Latn": "hechos no palabras",
        "deu_Latn": "taten nicht worte",
        "ita_Latn": "fatti non parole",
        "por_Latn": "acoes nao palavras"
      }
    },
    {
      "latin": "ad astra per aspera",
      "category": "phrase",
      "translations": {
        "eng_Latn": "to the stars through hardships",
        "fra_Latn": "vers les etoiles a travers les epreuves",
        "spa_Latn": "a las estrellas a traves de las dificultades",
        "deu_Latn": "zu den sternen durch muhen",
        "ita_Latn": "alle stelle attraverso le difficolta",
        "por_Latn": "as estrelas por meio das dificuldades"
      }
    },
    {
      "latin": "ad infinitum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "to infinity",
        "fra_Latn": "a l infini",
        "spa_Latn": "hasta el infinito",
        "deu_Latn": "bis ins unendliche",
        "ita_Latn": "all infinito",
        "por_Latn": "ao infinito"
      }
    },
    {
      "latin": "ad meliora",
      "category": "phrase",
      "translations": {
        "eng_Latn": "toward better things",
        "fra_Latn": "vers de meilleures choses",
        "spa_Latn": "hacia cosas mejores",
        "deu_Latn": "zu besseren dingen",
        "ita_Latn": "verso cose migliori",
        "por_Latn": "para coisas melhores"
      }
    },
    {
      "latin": "ad multos annos",
      "category": "phrase",
      "translations": {
        "eng_Latn": "for many years",
        "fra_Latn": "pour de nombreuses annees",
        "spa_Latn": "por muchos anos",
        "deu_Latn": "auf viele jahre",
        "ita_Latn": "per molti anni",
        "por_Latn": "por muitos anos"
      }
    },
    {
      "latin": "ad valorem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "according to value",
        "fra_Latn": "selon la valeur",
        "spa_Latn": "segun el valor",
        "deu_Latn": "nach wert",
        "ita_Latn": "secondo il valore",
        "por_Latn": "de acordo com o valor"
      }
    },
    {
      "latin": "ad victoriam",
      "category": "phrase",
      "translations": {
        "eng_Latn": "toward victory",
        "fra_Latn": "vers la victoire",
        "spa_Latn": "hacia la victoria",
        "deu_Latn": "zum sieg",
        "ita_Latn": "verso la vittoria",
        "por_Latn": "rumo a vitoria"
      }
    },
    {
      "latin": "alter ego",
      "category": "phrase",
      "translations": {
        "eng_Latn": "other self",
        "fra_Latn": "autre soi",
        "spa_Latn": "otro yo",
        "deu_Latn": "anderes ich",
        "ita_Latn": "altro io",
        "por_Latn": "outro eu"
      }
    },
    {
      "latin": "amicitia semper prodest",
      "category": "phrase",
      "translations": {
        "eng_Latn": "friendship is always useful",
        "fra_Latn": "l amitie est toujours utile",
        "spa_Latn": "la amistad siempre es util",
        "deu_Latn": "freundschaft ist immer nutzlich",
        "ita_Latn": "l amicizia e sempre utile",
        "por_Latn": "a amizade e sempre util"
      }
    },
    {
      "latin": "amor fati",
      "category": "phrase",
      "translations": {
        "eng_Latn": "love of fate",
        "fra_Latn": "amour du destin",
        "spa_Latn": "amor al destino",
        "deu_Latn": "liebe zum schicksal",
        "ita_Latn": "amore del destino",
        "por_Latn": "amor ao destino"
      }
    },
    {
      "latin": "amor patriae",
      "category": "phrase",
      "translations": {
        "eng_Latn": "love of country",
        "fra_Latn": "amour de la patrie",
        "spa_Latn": "amor a la patria",
        "deu_Latn": "liebe zum vaterland",
        "ita_Latn": "amore della patria",
        "por_Latn": "amor a patria"
      }
    },
    {
      "latin": "animus in consulendo liber",
      "category": "phrase",
      "translations": {
        "eng_Latn": "a mind free in deliberation",
        "fra_Latn": "un esprit libre dans la reflexion",
        "spa_Latn": "una mente libre al deliberar",
        "deu_Latn": "ein freier geist im beraten",
        "ita_Latn": "una mente libera nel deliberare",
        "por_Latn": "uma mente livre ao deliberar"
      }
    },
    {
      "latin": "annus horribilis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "horrible year",
        "fra_Latn": "annee terrible",
        "spa_Latn": "ano horrible",
        "deu_Latn": "schreckliches jahr",
        "ita_Latn": "anno terribile",
        "por_Latn": "ano horrivel"
      }
    },
    {
      "latin": "annus mirabilis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "wonderful year",
        "fra_Latn": "annee merveilleuse",
        "spa_Latn": "ano maravilloso",
        "deu_Latn": "wunderbares jahr",
        "ita_Latn": "anno meraviglioso",
        "por_Latn": "ano maravilhoso"
      }
    },
    {
      "latin": "aqua vitae",
      "category": "phrase",
      "translations": {
        "eng_Latn": "water of life",
        "fra_Latn": "eau de vie",
        "spa_Latn": "agua de vida",
        "deu_Latn": "wasser des lebens",
        "ita_Latn": "acqua della vita",
        "por_Latn": "agua da vida"
      }
    },
    {
      "latin": "ars gratia artis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "art for art s sake",
        "fra_Latn": "l art pour l art",
        "spa_Latn": "arte por el arte",
        "deu_Latn": "kunst um der kunst willen",
        "ita_Latn": "arte per l arte",
        "por_Latn": "arte pela arte"
      }
    },
    {
      "latin": "ars est celare artem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "art is to conceal art",
        "fra_Latn": "l art est de cacher l art",
        "spa_Latn": "el arte es ocultar el arte",
        "deu_Latn": "kunst ist es kunst zu verbergen",
        "ita_Latn": "l arte e nascondere l arte",
        "por_Latn": "a arte e ocultar a arte"
      }
    },
    {
      "latin": "audax at fidelis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "bold but faithful",
        "fra_Latn": "audacieux mais fidele",
        "spa_Latn": "audaz pero fiel",
        "deu_Latn": "kuhn aber treu",
        "ita_Latn": "audace ma fedele",
        "por_Latn": "ousado mas fiel"
      }
    },
    {
      "latin": "aurea mediocritas",
      "category": "phrase",
      "translations": {
        "eng_Latn": "golden mean",
        "fra_Latn": "juste milieu dore",
        "spa_Latn": "justo medio dorado",
        "deu_Latn": "goldene mitte",
        "ita_Latn": "giusto mezzo dorato",
        "por_Latn": "meio termo dourado"
      }
    },
    {
      "latin": "beatus ille",
      "category": "phrase",
      "translations": {
        "eng_Latn": "happy is that one",
        "fra_Latn": "heureux celui la",
        "spa_Latn": "feliz aquel",
        "deu_Latn": "glucklich ist jener",
        "ita_Latn": "beato colui",
        "por_Latn": "feliz aquele"
      }
    },
    {
      "latin": "bis dat qui cito dat",
      "category": "phrase",
      "translations": {
        "eng_Latn": "he gives twice who gives quickly",
        "fra_Latn": "il donne deux fois qui donne vite",
        "spa_Latn": "da dos veces quien da pronto",
        "deu_Latn": "doppelt gibt wer schnell gibt",
        "ita_Latn": "dona due volte chi dona presto",
        "por_Latn": "da duas vezes quem da cedo"
      }
    },
    {
      "latin": "bona fide",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in good faith",
        "fra_Latn": "de bonne foi",
        "spa_Latn": "de buena fe",
        "deu_Latn": "in gutem glauben",
        "ita_Latn": "in buona fede",
        "por_Latn": "de boa fe"
      }
    },
    {
      "latin": "bona fortuna",
      "category": "phrase",
      "translations": {
        "eng_Latn": "good fortune",
        "fra_Latn": "bonne fortune",
        "spa_Latn": "buena fortuna",
        "deu_Latn": "gutes gluck",
        "ita_Latn": "buona fortuna",
        "por_Latn": "boa fortuna"
      }
    },
    {
      "latin": "bonum commune communitatis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the common good of the community",
        "fra_Latn": "le bien commun de la communaute",
        "spa_Latn": "el bien comun de la comunidad",
        "deu_Latn": "das gemeinwohl der gemeinschaft",
        "ita_Latn": "il bene comune della comunita",
        "por_Latn": "o bem comum da comunidade"
      }
    },
    {
      "latin": "capta avis laeta est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the captured bird is joyful",
        "fra_Latn": "l oiseau capture est joyeux",
        "spa_Latn": "el pajaro capturado esta alegre",
        "deu_Latn": "der gefangene vogel ist froh",
        "ita_Latn": "l uccello catturato e lieto",
        "por_Latn": "o passaro capturado esta alegre"
      }
    },
    {
      "latin": "cave canem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "beware of the dog",
        "fra_Latn": "attention au chien",
        "spa_Latn": "cuidado con el perro",
        "deu_Latn": "hute dich vor dem hund",
        "ita_Latn": "attenti al cane",
        "por_Latn": "cuidado com o cao"
      }
    },
    {
      "latin": "caveat emptor",
      "category": "phrase",
      "translations": {
        "eng_Latn": "let the buyer beware",
        "fra_Latn": "que l acheteur prenne garde",
        "spa_Latn": "que el comprador tenga cuidado",
        "deu_Latn": "der kaufer moge sich huten",
        "ita_Latn": "stia attento il compratore",
        "por_Latn": "que o comprador se cuide"
      }
    },
    {
      "latin": "cedant arma togae",
      "category": "phrase",
      "translations": {
        "eng_Latn": "let arms yield to the toga",
        "fra_Latn": "que les armes cedent a la toge",
        "spa_Latn": "que las armas cedan a la toga",
        "deu_Latn": "die waffen sollen der toga weichen",
        "ita_Latn": "cedano le armi alla toga",
        "por_Latn": "que as armas cedam a toga"
      }
    },
    {
      "latin": "certum est quod certum reddi potest",
      "category": "phrase",
      "translations": {
        "eng_Latn": "that is certain which can be made certain",
        "fra_Latn": "est certain ce qui peut etre rendu certain",
        "spa_Latn": "es cierto lo que puede hacerse cierto",
        "deu_Latn": "gewiss ist was gewiss gemacht werden kann",
        "ita_Latn": "e certo cio che puo essere reso certo",
        "por_Latn": "e certo o que pode ser tornado certo"
      }
    },
    {
      "latin": "cogito ergo sum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "i think therefore i am",
        "fra_Latn": "je pense donc je suis",
        "spa_Latn": "pienso luego existo",
        "deu_Latn": "ich denke also bin ich",
        "ita_Latn": "penso dunque sono",
        "por_Latn": "penso logo existo"
      }
    },
    {
      "latin": "concordia parvae res crescunt",
      "category": "phrase",
      "translations": {
        "eng_Latn": "small things grow through harmony",
        "fra_Latn": "les petites choses grandissent par l harmonie",
        "spa_Latn": "las cosas pequenas crecen por la armonia",
        "deu_Latn": "kleine dinge wachsen durch eintracht",
        "ita_Latn": "le piccole cose crescono con l armonia",
        "por_Latn": "as pequenas coisas crescem pela harmonia"
      }
    },
    {
      "latin": "consilio et animis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "by wisdom and courage",
        "fra_Latn": "par la sagesse et le courage",
        "spa_Latn": "con sabiduria y valor",
        "deu_Latn": "durch klugheit und mut",
        "ita_Latn": "con saggezza e coraggio",
        "por_Latn": "com sabedoria e coragem"
      }
    },
    {
      "latin": "consuetudo est altera natura",
      "category": "phrase",
      "translations": {
        "eng_Latn": "habit is a second nature",
        "fra_Latn": "l habitude est une seconde nature",
        "spa_Latn": "la costumbre es una segunda naturaleza",
        "deu_Latn": "gewohnheit ist eine zweite natur",
        "ita_Latn": "l abitudine e una seconda natura",
        "por_Latn": "o habito e uma segunda natureza"
      }
    },
    {
      "latin": "cor ad cor loquitur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "heart speaks to heart",
        "fra_Latn": "le coeur parle au coeur",
        "spa_Latn": "el corazon habla al corazon",
        "deu_Latn": "herz spricht zu herz",
        "ita_Latn": "il cuore parla al cuore",
        "por_Latn": "o coracao fala ao coracao"
      }
    },
    {
      "latin": "cor unum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "one heart",
        "fra_Latn": "un seul coeur",
        "spa_Latn": "un solo corazon",
        "deu_Latn": "ein herz",
        "ita_Latn": "un solo cuore",
        "por_Latn": "um so coracao"
      }
    },
    {
      "latin": "cura ut valeas",
      "category": "phrase",
      "translations": {
        "eng_Latn": "take care that you are well",
        "fra_Latn": "prends soin de toi",
        "spa_Latn": "cuidate",
        "deu_Latn": "sorge gut fur dich",
        "ita_Latn": "abbi cura di te",
        "por_Latn": "cuida de ti"
      }
    },
    {
      "latin": "curriculum vitae",
      "category": "phrase",
      "translations": {
        "eng_Latn": "course of life",
        "fra_Latn": "cours de la vie",
        "spa_Latn": "curso de la vida",
        "deu_Latn": "lebenslauf",
        "ita_Latn": "corso della vita",
        "por_Latn": "curso da vida"
      }
    },
    {
      "latin": "de facto",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in fact",
        "fra_Latn": "de fait",
        "spa_Latn": "de hecho",
        "deu_Latn": "tatsachlich",
        "ita_Latn": "di fatto",
        "por_Latn": "de fato"
      }
    },
    {
      "latin": "de gustibus non est disputandum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "there is no disputing about tastes",
        "fra_Latn": "des gouts on ne discute pas",
        "spa_Latn": "sobre gustos no hay disputa",
        "deu_Latn": "uber geschmack lasst sich nicht streiten",
        "ita_Latn": "dei gusti non si discute",
        "por_Latn": "sobre gostos nao se discute"
      }
    },
    {
      "latin": "de integro",
      "category": "phrase",
      "translations": {
        "eng_Latn": "anew",
        "fra_Latn": "a nouveau",
        "spa_Latn": "de nuevo",
        "deu_Latn": "von neuem",
        "ita_Latn": "di nuovo",
        "por_Latn": "de novo"
      }
    },
    {
      "latin": "de jure",
      "category": "phrase",
      "translations": {
        "eng_Latn": "by law",
        "fra_Latn": "de droit",
        "spa_Latn": "de derecho",
        "deu_Latn": "rechtlich",
        "ita_Latn": "di diritto",
        "por_Latn": "de direito"
      }
    },
    {
      "latin": "de mortuis nil nisi bonum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "say nothing but good of the dead",
        "fra_Latn": "ne rien dire que du bien des morts",
        "spa_Latn": "no decir nada sino bien de los muertos",
        "deu_Latn": "uber tote nur gutes sagen",
        "ita_Latn": "dire solo bene dei morti",
        "por_Latn": "dizer apenas o bem dos mortos"
      }
    },
    {
      "latin": "de novo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "again from the beginning",
        "fra_Latn": "de nouveau depuis le debut",
        "spa_Latn": "de nuevo desde el principio",
        "deu_Latn": "wieder von vorn",
        "ita_Latn": "di nuovo dall inizio",
        "por_Latn": "de novo desde o inicio"
      }
    },
    {
      "latin": "de omnibus dubitandum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "everything must be doubted",
        "fra_Latn": "il faut douter de tout",
        "spa_Latn": "hay que dudar de todo",
        "deu_Latn": "an allem ist zu zweifeln",
        "ita_Latn": "bisogna dubitare di tutto",
        "por_Latn": "e preciso duvidar de tudo"
      }
    },
    {
      "latin": "deus ex machina",
      "category": "phrase",
      "translations": {
        "eng_Latn": "god from the machine",
        "fra_Latn": "dieu sorti de la machine",
        "spa_Latn": "dios desde la maquina",
        "deu_Latn": "gott aus der maschine",
        "ita_Latn": "dio dalla macchina",
        "por_Latn": "deus vindo da maquina"
      }
    },
    {
      "latin": "diligentia maximum etiam mediocris ingenii subsidium",
      "category": "phrase",
      "translations": {
        "eng_Latn": "diligence is the greatest aid even to average talent",
        "fra_Latn": "la diligence est la plus grande aide meme au talent moyen",
        "spa_Latn": "la diligencia es la mayor ayuda incluso al talento medio",
        "deu_Latn": "fleiss ist die grosste hilfe selbst fur mittlere begabung",
        "ita_Latn": "la diligenza e il massimo aiuto anche al talento medio",
        "por_Latn": "a diligencia e a maior ajuda ate para o talento medio"
      }
    },
    {
      "latin": "divide et impera",
      "category": "phrase",
      "translations": {
        "eng_Latn": "divide and rule",
        "fra_Latn": "diviser pour regner",
        "spa_Latn": "divide y gobierna",
        "deu_Latn": "teile und herrsche",
        "ita_Latn": "dividi e governa",
        "por_Latn": "divide e governa"
      }
    },
    {
      "latin": "do ut des",
      "category": "phrase",
      "translations": {
        "eng_Latn": "i give so that you may give",
        "fra_Latn": "je donne pour que tu donnes",
        "spa_Latn": "doy para que des",
        "deu_Latn": "ich gebe damit du gibst",
        "ita_Latn": "do affinche tu dia",
        "por_Latn": "dou para que tu des"
      }
    },
    {
      "latin": "docendo discimus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "by teaching we learn",
        "fra_Latn": "en enseignant nous apprenons",
        "spa_Latn": "ensenando aprendemos",
        "deu_Latn": "durch lehren lernen wir",
        "ita_Latn": "insegnando impariamo",
        "por_Latn": "ensinando aprendemos"
      }
    },
    {
      "latin": "dum vita est spes est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "while there is life there is hope",
        "fra_Latn": "tant qu il y a de la vie il y a de l espoir",
        "spa_Latn": "mientras hay vida hay esperanza",
        "deu_Latn": "solange leben ist ist hoffnung",
        "ita_Latn": "finche c e vita c e speranza",
        "por_Latn": "enquanto ha vida ha esperanca"
      }
    },
    {
      "latin": "ecce homo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "behold the man",
        "fra_Latn": "voici l homme",
        "spa_Latn": "he aqui el hombre",
        "deu_Latn": "seht den menschen",
        "ita_Latn": "ecco l uomo",
        "por_Latn": "eis o homem"
      }
    },
    {
      "latin": "editio princeps",
      "category": "phrase",
      "translations": {
        "eng_Latn": "first edition",
        "fra_Latn": "premiere edition",
        "spa_Latn": "primera edicion",
        "deu_Latn": "erste ausgabe",
        "ita_Latn": "prima edizione",
        "por_Latn": "primeira edicao"
      }
    },
    {
      "latin": "ensis in pace levis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the sword is light in peace",
        "fra_Latn": "l epee est legere en paix",
        "spa_Latn": "la espada es ligera en paz",
        "deu_Latn": "das schwert ist im frieden leicht",
        "ita_Latn": "la spada e leggera in pace",
        "por_Latn": "a espada e leve na paz"
      }
    },
    {
      "latin": "esto perpetua",
      "category": "phrase",
      "translations": {
        "eng_Latn": "let it be perpetual",
        "fra_Latn": "que cela soit perpetuel",
        "spa_Latn": "que sea perpetuo",
        "deu_Latn": "es moge ewig sein",
        "ita_Latn": "sia perpetuo",
        "por_Latn": "que seja perpetuo"
      }
    },
    {
      "latin": "ex abundantia cordis os loquitur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "out of the abundance of the heart the mouth speaks",
        "fra_Latn": "de l abondance du coeur la bouche parle",
        "spa_Latn": "de la abundancia del corazon habla la boca",
        "deu_Latn": "wovon das herz voll ist davon spricht der mund",
        "ita_Latn": "dall abbondanza del cuore parla la bocca",
        "por_Latn": "da abundancia do coracao fala a boca"
      }
    },
    {
      "latin": "ex animo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "from the heart",
        "fra_Latn": "du coeur",
        "spa_Latn": "de corazon",
        "deu_Latn": "von herzen",
        "ita_Latn": "di cuore",
        "por_Latn": "de coracao"
      }
    },
    {
      "latin": "ex cathedra",
      "category": "phrase",
      "translations": {
        "eng_Latn": "from the chair of authority",
        "fra_Latn": "depuis la chaire d autorite",
        "spa_Latn": "desde la catedra de autoridad",
        "deu_Latn": "von der lehrkanzel aus",
        "ita_Latn": "dalla cattedra di autorita",
        "por_Latn": "da cadeira de autoridade"
      }
    },
    {
      "latin": "ex nihilo nihil fit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "nothing comes from nothing",
        "fra_Latn": "rien ne vient de rien",
        "spa_Latn": "nada viene de la nada",
        "deu_Latn": "von nichts kommt nichts",
        "ita_Latn": "nulla viene dal nulla",
        "por_Latn": "nada vem do nada"
      }
    },
    {
      "latin": "ex officio",
      "category": "phrase",
      "translations": {
        "eng_Latn": "by virtue of office",
        "fra_Latn": "en vertu de la fonction",
        "spa_Latn": "por virtud del cargo",
        "deu_Latn": "kraft amtes",
        "ita_Latn": "in virtu dell ufficio",
        "por_Latn": "em virtude do cargo"
      }
    },
    {
      "latin": "ex silentio",
      "category": "phrase",
      "translations": {
        "eng_Latn": "from silence",
        "fra_Latn": "du silence",
        "spa_Latn": "desde el silencio",
        "deu_Latn": "aus dem schweigen",
        "ita_Latn": "dal silenzio",
        "por_Latn": "do silencio"
      }
    },
    {
      "latin": "ex tempore",
      "category": "phrase",
      "translations": {
        "eng_Latn": "without preparation",
        "fra_Latn": "sans preparation",
        "spa_Latn": "sin preparacion",
        "deu_Latn": "ohne vorbereitung",
        "ita_Latn": "senza preparazione",
        "por_Latn": "sem preparacao"
      }
    },
    {
      "latin": "facta sunt potentiora verbis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "deeds are stronger than words",
        "fra_Latn": "les actes sont plus forts que les mots",
        "spa_Latn": "los hechos son mas fuertes que las palabras",
        "deu_Latn": "taten sind starker als worte",
        "ita_Latn": "i fatti sono piu forti delle parole",
        "por_Latn": "os atos sao mais fortes que as palavras"
      }
    },
    {
      "latin": "fama volat",
      "category": "phrase",
      "translations": {
        "eng_Latn": "rumor flies",
        "fra_Latn": "la rumeur vole",
        "spa_Latn": "el rumor vuela",
        "deu_Latn": "das gerucht fliegt",
        "ita_Latn": "la fama vola",
        "por_Latn": "o rumor voa"
      }
    },
    {
      "latin": "festina lente sapienter",
      "category": "phrase",
      "translations": {
        "eng_Latn": "make haste slowly and wisely",
        "fra_Latn": "hate toi lentement et sagement",
        "spa_Latn": "apresurate despacio y sabiamente",
        "deu_Latn": "eile langsam und weise",
        "ita_Latn": "affrettati lentamente e con saggezza",
        "por_Latn": "apressa te devagar e com sabedoria"
      }
    },
    {
      "latin": "fiat iustitia ruat caelum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "let justice be done though the heavens fall",
        "fra_Latn": "que justice soit faite meme si le ciel tombe",
        "spa_Latn": "que se haga justicia aunque caiga el cielo",
        "deu_Latn": "es geschehe gerechtigkeit und wenn der himmel fallt",
        "ita_Latn": "sia fatta giustizia anche se il cielo crolla",
        "por_Latn": "faca se justica ainda que o ceu caia"
      }
    },
    {
      "latin": "fiat lux",
      "category": "phrase",
      "translations": {
        "eng_Latn": "let there be light",
        "fra_Latn": "que la lumiere soit",
        "spa_Latn": "hagase la luz",
        "deu_Latn": "es werde licht",
        "ita_Latn": "sia fatta la luce",
        "por_Latn": "haja luz"
      }
    },
    {
      "latin": "fides et fortitudo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "faith and courage",
        "fra_Latn": "foi et courage",
        "spa_Latn": "fe y valor",
        "deu_Latn": "glaube und mut",
        "ita_Latn": "fede e coraggio",
        "por_Latn": "fe e coragem"
      }
    },
    {
      "latin": "fides quaerens intellectum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "faith seeking understanding",
        "fra_Latn": "la foi cherchant l intelligence",
        "spa_Latn": "la fe que busca entendimiento",
        "deu_Latn": "glaube der verstandnis sucht",
        "ita_Latn": "fede che cerca comprensione",
        "por_Latn": "fe buscando entendimento"
      }
    },
    {
      "latin": "finis coronat opus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the end crowns the work",
        "fra_Latn": "la fin couronne l oeuvre",
        "spa_Latn": "el fin corona la obra",
        "deu_Latn": "das ende kronet das werk",
        "ita_Latn": "la fine incorona l opera",
        "por_Latn": "o fim coroa a obra"
      }
    },
    {
      "latin": "fortes fortuna adiuvat",
      "category": "phrase",
      "translations": {
        "eng_Latn": "fortune helps the brave",
        "fra_Latn": "la fortune aide les braves",
        "spa_Latn": "la fortuna ayuda a los valientes",
        "deu_Latn": "das gluck hilft den tapferen",
        "ita_Latn": "la fortuna aiuta i forti",
        "por_Latn": "a fortuna ajuda os corajosos"
      }
    },
    {
      "latin": "gaudeamus igitur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "therefore let us rejoice",
        "fra_Latn": "rejouissons nous donc",
        "spa_Latn": "alegremonos entonces",
        "deu_Latn": "also lasst uns froh sein",
        "ita_Latn": "rallegriamoci dunque",
        "por_Latn": "alegremo nos portanto"
      }
    },
    {
      "latin": "gloria in excelsis deo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "glory to god in the highest",
        "fra_Latn": "gloire a dieu au plus haut des cieux",
        "spa_Latn": "gloria a dios en las alturas",
        "deu_Latn": "ehre sei gott in der hohe",
        "ita_Latn": "gloria a dio nell alto dei cieli",
        "por_Latn": "gloria a deus nas alturas"
      }
    },
    {
      "latin": "gnothi te ipsum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "know thyself",
        "fra_Latn": "connais toi toi meme",
        "spa_Latn": "conocete a ti mismo",
        "deu_Latn": "erkenne dich selbst",
        "ita_Latn": "conosci te stesso",
        "por_Latn": "conhece te a ti mesmo"
      }
    },
    {
      "latin": "habemus papam",
      "category": "phrase",
      "translations": {
        "eng_Latn": "we have a pope",
        "fra_Latn": "nous avons un pape",
        "spa_Latn": "tenemos papa",
        "deu_Latn": "wir haben einen papst",
        "ita_Latn": "abbiamo un papa",
        "por_Latn": "temos um papa"
      }
    },
    {
      "latin": "habent sua fata libelli",
      "category": "phrase",
      "translations": {
        "eng_Latn": "books have their own destinies",
        "fra_Latn": "les livres ont leur propre destin",
        "spa_Latn": "los libros tienen su propio destino",
        "deu_Latn": "bucher haben ihr eigenes schicksal",
        "ita_Latn": "i libri hanno il loro destino",
        "por_Latn": "os livros tem o seu proprio destino"
      }
    },
    {
      "latin": "hic et nunc",
      "category": "phrase",
      "translations": {
        "eng_Latn": "here and now",
        "fra_Latn": "ici et maintenant",
        "spa_Latn": "aqui y ahora",
        "deu_Latn": "hier und jetzt",
        "ita_Latn": "qui e ora",
        "por_Latn": "aqui e agora"
      }
    },
    {
      "latin": "hic sunt dracones",
      "category": "phrase",
      "translations": {
        "eng_Latn": "here be dragons",
        "fra_Latn": "ici sont les dragons",
        "spa_Latn": "aqui hay dragones",
        "deu_Latn": "hier sind drachen",
        "ita_Latn": "qui ci sono draghi",
        "por_Latn": "aqui ha dragoes"
      }
    },
    {
      "latin": "hodie mihi cras tibi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "today me tomorrow you",
        "fra_Latn": "aujourd hui moi demain toi",
        "spa_Latn": "hoy yo manana tu",
        "deu_Latn": "heute ich morgen du",
        "ita_Latn": "oggi a me domani a te",
        "por_Latn": "hoje eu amanha tu"
      }
    },
    {
      "latin": "homo faber",
      "category": "phrase",
      "translations": {
        "eng_Latn": "man the maker",
        "fra_Latn": "l homme createur",
        "spa_Latn": "el hombre creador",
        "deu_Latn": "der schaffende mensch",
        "ita_Latn": "l uomo creatore",
        "por_Latn": "o homem criador"
      }
    },
    {
      "latin": "homo homini lupus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "man is a wolf to man",
        "fra_Latn": "l homme est un loup pour l homme",
        "spa_Latn": "el hombre es un lobo para el hombre",
        "deu_Latn": "der mensch ist dem menschen ein wolf",
        "ita_Latn": "l uomo e un lupo per l uomo",
        "por_Latn": "o homem e um lobo para o homem"
      }
    },
    {
      "latin": "homo sum humani nihil a me alienum puto",
      "category": "phrase",
      "translations": {
        "eng_Latn": "i am human and think nothing human is alien to me",
        "fra_Latn": "je suis humain et rien de ce qui est humain ne m est etranger",
        "spa_Latn": "soy humano y nada humano me es ajeno",
        "deu_Latn": "ich bin mensch und halte nichts menschliches fur mir fremd",
        "ita_Latn": "sono uomo e nulla di umano ritengo estraneo a me",
        "por_Latn": "sou humano e nada do que e humano me e estranho"
      }
    },
    {
      "latin": "honor virtutis praemium",
      "category": "phrase",
      "translations": {
        "eng_Latn": "honor is the reward of virtue",
        "fra_Latn": "l honneur est la recompense de la vertu",
        "spa_Latn": "el honor es la recompensa de la virtud",
        "deu_Latn": "ehre ist der lohn der tugend",
        "ita_Latn": "l onore e il premio della virtu",
        "por_Latn": "a honra e o premio da virtude"
      }
    },
    {
      "latin": "hora fugit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the hour flies",
        "fra_Latn": "l heure s enfuit",
        "spa_Latn": "la hora vuela",
        "deu_Latn": "die stunde fliegt",
        "ita_Latn": "l ora fugge",
        "por_Latn": "a hora foge"
      }
    },
    {
      "latin": "humanitas",
      "category": "word",
      "translations": {
        "eng_Latn": "humanity",
        "fra_Latn": "humanite",
        "spa_Latn": "humanidad",
        "deu_Latn": "menschlichkeit",
        "ita_Latn": "umanita",
        "por_Latn": "humanidade"
      }
    },
    {
      "latin": "ibidem",
      "category": "word",
      "translations": {
        "eng_Latn": "in the same place",
        "fra_Latn": "au meme endroit",
        "spa_Latn": "en el mismo lugar",
        "deu_Latn": "am selben ort",
        "ita_Latn": "nello stesso luogo",
        "por_Latn": "no mesmo lugar"
      }
    },
    {
      "latin": "idem",
      "category": "word",
      "translations": {
        "eng_Latn": "the same",
        "fra_Latn": "le meme",
        "spa_Latn": "lo mismo",
        "deu_Latn": "dasselbe",
        "ita_Latn": "lo stesso",
        "por_Latn": "o mesmo"
      }
    },
    {
      "latin": "ignotum per ignotius",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the unknown by the more unknown",
        "fra_Latn": "l inconnu par le plus inconnu",
        "spa_Latn": "lo desconocido por lo mas desconocido",
        "deu_Latn": "das unbekannte durch das noch unbekanntere",
        "ita_Latn": "l ignoto per il piu ignoto",
        "por_Latn": "o desconhecido pelo mais desconhecido"
      }
    },
    {
      "latin": "imago dei",
      "category": "phrase",
      "translations": {
        "eng_Latn": "image of god",
        "fra_Latn": "image de dieu",
        "spa_Latn": "imagen de dios",
        "deu_Latn": "ebenbild gottes",
        "ita_Latn": "immagine di dio",
        "por_Latn": "imagem de deus"
      }
    },
    {
      "latin": "imperium in imperio",
      "category": "phrase",
      "translations": {
        "eng_Latn": "an empire within an empire",
        "fra_Latn": "un empire dans l empire",
        "spa_Latn": "un imperio dentro de un imperio",
        "deu_Latn": "ein reich im reich",
        "ita_Latn": "un impero nell impero",
        "por_Latn": "um imperio dentro do imperio"
      }
    },
    {
      "latin": "in absentia",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in absence",
        "fra_Latn": "en absence",
        "spa_Latn": "en ausencia",
        "deu_Latn": "in abwesenheit",
        "ita_Latn": "in assenza",
        "por_Latn": "em ausencia"
      }
    },
    {
      "latin": "in aeternum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "forever",
        "fra_Latn": "pour toujours",
        "spa_Latn": "para siempre",
        "deu_Latn": "fur ewig",
        "ita_Latn": "per sempre",
        "por_Latn": "para sempre"
      }
    },
    {
      "latin": "in articulo mortis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "at the point of death",
        "fra_Latn": "a l article de la mort",
        "spa_Latn": "en articulo de muerte",
        "deu_Latn": "im angesicht des todes",
        "ita_Latn": "in punto di morte",
        "por_Latn": "a beira da morte"
      }
    },
    {
      "latin": "in cauda venenum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the poison is in the tail",
        "fra_Latn": "le poison est dans la queue",
        "spa_Latn": "el veneno esta en la cola",
        "deu_Latn": "das gift steckt im schwanz",
        "ita_Latn": "il veleno e nella coda",
        "por_Latn": "o veneno esta na cauda"
      }
    },
    {
      "latin": "in dubio pro reo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "when in doubt for the accused",
        "fra_Latn": "en cas de doute en faveur de l accuse",
        "spa_Latn": "en caso de duda a favor del acusado",
        "deu_Latn": "im zweifel fur den angeklagten",
        "ita_Latn": "nel dubbio a favore dell imputato",
        "por_Latn": "na duvida a favor do reu"
      }
    },
    {
      "latin": "in extremis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "at the very end",
        "fra_Latn": "a l extremite",
        "spa_Latn": "en el ultimo momento",
        "deu_Latn": "im letzten augenblick",
        "ita_Latn": "all estremo",
        "por_Latn": "no ultimo instante"
      }
    },
    {
      "latin": "in flore",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in bloom",
        "fra_Latn": "en fleur",
        "spa_Latn": "en flor",
        "deu_Latn": "in blute",
        "ita_Latn": "in fiore",
        "por_Latn": "em flor"
      }
    },
    {
      "latin": "in hoc signo vinces",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in this sign you will conquer",
        "fra_Latn": "par ce signe tu vaincras",
        "spa_Latn": "con este signo venceras",
        "deu_Latn": "in diesem zeichen wirst du siegen",
        "ita_Latn": "con questo segno vincerai",
        "por_Latn": "com este sinal venceras"
      }
    },
    {
      "latin": "in medias res",
      "category": "phrase",
      "translations": {
        "eng_Latn": "into the middle of things",
        "fra_Latn": "au milieu des choses",
        "spa_Latn": "en medio de las cosas",
        "deu_Latn": "mitten in die dinge",
        "ita_Latn": "nel mezzo delle cose",
        "por_Latn": "no meio das coisas"
      }
    },
    {
      "latin": "in memoriam",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in memory",
        "fra_Latn": "en memoire",
        "spa_Latn": "en memoria",
        "deu_Latn": "zum gedenken",
        "ita_Latn": "in memoria",
        "por_Latn": "em memoria"
      }
    },
    {
      "latin": "in omnia paratus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "prepared for all things",
        "fra_Latn": "pret pour toutes choses",
        "spa_Latn": "preparado para todo",
        "deu_Latn": "auf alles vorbereitet",
        "ita_Latn": "pronto a tutto",
        "por_Latn": "preparado para tudo"
      }
    },
    {
      "latin": "in pari materia",
      "category": "phrase",
      "translations": {
        "eng_Latn": "on the same subject",
        "fra_Latn": "sur la meme matiere",
        "spa_Latn": "sobre la misma materia",
        "deu_Latn": "in derselben sache",
        "ita_Latn": "sulla stessa materia",
        "por_Latn": "sobre a mesma materia"
      }
    },
    {
      "latin": "in perpetuum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "for ever",
        "fra_Latn": "a perpetuite",
        "spa_Latn": "para siempre",
        "deu_Latn": "auf ewig",
        "ita_Latn": "in perpetuo",
        "por_Latn": "para sempre"
      }
    },
    {
      "latin": "in propria persona",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in one s own person",
        "fra_Latn": "en personne",
        "spa_Latn": "en persona propia",
        "deu_Latn": "in eigener person",
        "ita_Latn": "in persona propria",
        "por_Latn": "em propria pessoa"
      }
    },
    {
      "latin": "in situ",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in position",
        "fra_Latn": "sur place",
        "spa_Latn": "en su lugar",
        "deu_Latn": "an ort und stelle",
        "ita_Latn": "in situ",
        "por_Latn": "no local"
      }
    },
    {
      "latin": "in somnis veritas",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in dreams there is truth",
        "fra_Latn": "dans les reves il y a la verite",
        "spa_Latn": "en los suenos esta la verdad",
        "deu_Latn": "in traumen liegt wahrheit",
        "ita_Latn": "nei sogni c e verita",
        "por_Latn": "nos sonhos esta a verdade"
      }
    },
    {
      "latin": "in toto",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in total",
        "fra_Latn": "dans son ensemble",
        "spa_Latn": "en total",
        "deu_Latn": "im ganzen",
        "ita_Latn": "nel totale",
        "por_Latn": "no total"
      }
    },
    {
      "latin": "in utrumque paratus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "prepared for either outcome",
        "fra_Latn": "pret a l un ou l autre resultat",
        "spa_Latn": "preparado para cualquiera de los dos resultados",
        "deu_Latn": "auf beide moglichkeiten vorbereitet",
        "ita_Latn": "pronto a entrambi gli esiti",
        "por_Latn": "preparado para qualquer dos dois resultados"
      }
    },
    {
      "latin": "in vino sanitas",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in wine there is health",
        "fra_Latn": "dans le vin il y a la sante",
        "spa_Latn": "en el vino hay salud",
        "deu_Latn": "im wein liegt gesundheit",
        "ita_Latn": "nel vino c e salute",
        "por_Latn": "no vinho ha saude"
      }
    },
    {
      "latin": "inveniam viam aut faciam",
      "category": "phrase",
      "translations": {
        "eng_Latn": "i shall find a way or make one",
        "fra_Latn": "je trouverai un chemin ou j en ferai un",
        "spa_Latn": "encontrare un camino o lo hare",
        "deu_Latn": "ich werde einen weg finden oder einen machen",
        "ita_Latn": "trovero una via o la faro",
        "por_Latn": "encontrarei um caminho ou o farei"
      }
    },
    {
      "latin": "ipsa scientia potestas est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "knowledge itself is power",
        "fra_Latn": "la connaissance elle meme est pouvoir",
        "spa_Latn": "el conocimiento mismo es poder",
        "deu_Latn": "wissen selbst ist macht",
        "ita_Latn": "la conoscenza stessa e potere",
        "por_Latn": "o proprio conhecimento e poder"
      }
    },
    {
      "latin": "ipse dixit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "he himself said it",
        "fra_Latn": "il l a dit lui meme",
        "spa_Latn": "el mismo lo dijo",
        "deu_Latn": "er selbst hat es gesagt",
        "ita_Latn": "lo ha detto lui stesso",
        "por_Latn": "ele proprio o disse"
      }
    },
    {
      "latin": "ira furor brevis est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "anger is a brief madness",
        "fra_Latn": "la colere est une breve folie",
        "spa_Latn": "la ira es una breve locura",
        "deu_Latn": "zorn ist ein kurzer wahnsinn",
        "ita_Latn": "l ira e una breve follia",
        "por_Latn": "a ira e uma breve loucura"
      }
    },
    {
      "latin": "labor omnia superat",
      "category": "phrase",
      "translations": {
        "eng_Latn": "work overcomes all things",
        "fra_Latn": "le travail surmonte tout",
        "spa_Latn": "el trabajo supera todo",
        "deu_Latn": "arbeit uberwindet alles",
        "ita_Latn": "il lavoro supera ogni cosa",
        "por_Latn": "o trabalho supera tudo"
      }
    },
    {
      "latin": "lapsus linguae",
      "category": "phrase",
      "translations": {
        "eng_Latn": "slip of the tongue",
        "fra_Latn": "lapsus de la langue",
        "spa_Latn": "lapsus de la lengua",
        "deu_Latn": "versprecher",
        "ita_Latn": "lapsus della lingua",
        "por_Latn": "lapso de lingua"
      }
    },
    {
      "latin": "laus deo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "praise be to god",
        "fra_Latn": "louange a dieu",
        "spa_Latn": "alabado sea dios",
        "deu_Latn": "gott sei gelobt",
        "ita_Latn": "lode a dio",
        "por_Latn": "louvado seja deus"
      }
    },
    {
      "latin": "lectio brevior potior",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the shorter reading is stronger",
        "fra_Latn": "la lecture la plus breve est plus forte",
        "spa_Latn": "la lectura mas breve es mas fuerte",
        "deu_Latn": "die kurzere lesart ist starker",
        "ita_Latn": "la lezione piu breve e piu forte",
        "por_Latn": "a leitura mais breve e mais forte"
      }
    },
    {
      "latin": "lege artis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "according to the rules of the art",
        "fra_Latn": "selon les regles de l art",
        "spa_Latn": "segun las reglas del arte",
        "deu_Latn": "nach den regeln der kunst",
        "ita_Latn": "secondo le regole dell arte",
        "por_Latn": "segundo as regras da arte"
      }
    },
    {
      "latin": "lex orandi lex credendi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the law of prayer is the law of belief",
        "fra_Latn": "la loi de la priere est la loi de la foi",
        "spa_Latn": "la ley de la oracion es la ley de la creencia",
        "deu_Latn": "das gesetz des betens ist das gesetz des glaubens",
        "ita_Latn": "la legge della preghiera e la legge della fede",
        "por_Latn": "a lei da oracao e a lei da crenca"
      }
    },
    {
      "latin": "lex parsimoniae",
      "category": "phrase",
      "translations": {
        "eng_Latn": "law of parsimony",
        "fra_Latn": "loi de parcimonie",
        "spa_Latn": "ley de parsimonia",
        "deu_Latn": "gesetz der sparsamkeit",
        "ita_Latn": "legge della parsimonia",
        "por_Latn": "lei da parcimonia"
      }
    },
    {
      "latin": "lex talionis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "law of retaliation",
        "fra_Latn": "loi du talion",
        "spa_Latn": "ley del talion",
        "deu_Latn": "vergeltungsgesetz",
        "ita_Latn": "legge del taglione",
        "por_Latn": "lei de talião"
      }
    },
    {
      "latin": "libertas perfundet omnia luce",
      "category": "phrase",
      "translations": {
        "eng_Latn": "liberty will flood all things with light",
        "fra_Latn": "la liberte inondera tout de lumiere",
        "spa_Latn": "la libertad inundara todo con luz",
        "deu_Latn": "freiheit wird alles mit licht uberfluten",
        "ita_Latn": "la liberta riempira tutto di luce",
        "por_Latn": "a liberdade inundara tudo de luz"
      }
    },
    {
      "latin": "locus classicus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "classic passage",
        "fra_Latn": "passage classique",
        "spa_Latn": "pasaje clasico",
        "deu_Latn": "klassische stelle",
        "ita_Latn": "passo classico",
        "por_Latn": "passagem classica"
      }
    },
    {
      "latin": "locus standi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "place of standing",
        "fra_Latn": "qualite pour agir",
        "spa_Latn": "legitimacion para actuar",
        "deu_Latn": "klagebefugnis",
        "ita_Latn": "legittimazione ad agire",
        "por_Latn": "legitimidade para agir"
      }
    },
    {
      "latin": "magna cum laude",
      "category": "phrase",
      "translations": {
        "eng_Latn": "with great praise",
        "fra_Latn": "avec grande louange",
        "spa_Latn": "con gran elogio",
        "deu_Latn": "mit grossem lob",
        "ita_Latn": "con grande lode",
        "por_Latn": "com grande louvor"
      }
    },
    {
      "latin": "magnum opus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "great work",
        "fra_Latn": "grande oeuvre",
        "spa_Latn": "gran obra",
        "deu_Latn": "grosses werk",
        "ita_Latn": "grande opera",
        "por_Latn": "grande obra"
      }
    },
    {
      "latin": "mater semper certa est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the mother is always certain",
        "fra_Latn": "la mere est toujours certaine",
        "spa_Latn": "la madre siempre es cierta",
        "deu_Latn": "die mutter ist immer gewiss",
        "ita_Latn": "la madre e sempre certa",
        "por_Latn": "a mae e sempre certa"
      }
    },
    {
      "latin": "mea culpa",
      "category": "phrase",
      "translations": {
        "eng_Latn": "my fault",
        "fra_Latn": "ma faute",
        "spa_Latn": "mi culpa",
        "deu_Latn": "meine schuld",
        "ita_Latn": "mia colpa",
        "por_Latn": "minha culpa"
      }
    },
    {
      "latin": "medice cura te ipsum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "physician heal thyself",
        "fra_Latn": "medecin gueris toi toi meme",
        "spa_Latn": "medico curate a ti mismo",
        "deu_Latn": "arzt heile dich selbst",
        "ita_Latn": "medico cura te stesso",
        "por_Latn": "medico cura te a ti mesmo"
      }
    },
    {
      "latin": "memoria",
      "category": "word",
      "translations": {
        "eng_Latn": "memory",
        "fra_Latn": "memoire",
        "spa_Latn": "memoria",
        "deu_Latn": "erinnerung",
        "ita_Latn": "memoria",
        "por_Latn": "memoria"
      }
    },
    {
      "latin": "mens agitat molem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "mind moves matter",
        "fra_Latn": "l esprit meut la matiere",
        "spa_Latn": "la mente mueve la materia",
        "deu_Latn": "der geist bewegt die materie",
        "ita_Latn": "la mente muove la materia",
        "por_Latn": "a mente move a materia"
      }
    },
    {
      "latin": "mens sana in corpore sano",
      "category": "phrase",
      "translations": {
        "eng_Latn": "a sound mind in a sound body",
        "fra_Latn": "un esprit sain dans un corps sain",
        "spa_Latn": "mente sana en cuerpo sano",
        "deu_Latn": "ein gesunder geist in einem gesunden korper",
        "ita_Latn": "mente sana in corpo sano",
        "por_Latn": "mente sã em corpo são"
      }
    },
    {
      "latin": "mens sibi conscia recti",
      "category": "phrase",
      "translations": {
        "eng_Latn": "a mind conscious of right",
        "fra_Latn": "un esprit conscient du bien",
        "spa_Latn": "una mente consciente de lo correcto",
        "deu_Latn": "ein des rechten bewusstes gemut",
        "ita_Latn": "una mente consapevole del giusto",
        "por_Latn": "uma mente consciente do correto"
      }
    },
    {
      "latin": "mirabile dictu",
      "category": "phrase",
      "translations": {
        "eng_Latn": "wonderful to say",
        "fra_Latn": "merveilleux a dire",
        "spa_Latn": "maravilloso de decir",
        "deu_Latn": "wunderbar zu sagen",
        "ita_Latn": "meraviglioso a dirsi",
        "por_Latn": "maravilhoso de dizer"
      }
    },
    {
      "latin": "mirabile visu",
      "category": "phrase",
      "translations": {
        "eng_Latn": "wonderful to behold",
        "fra_Latn": "merveilleux a voir",
        "spa_Latn": "maravilloso de ver",
        "deu_Latn": "wunderbar anzusehen",
        "ita_Latn": "meraviglioso a vedersi",
        "por_Latn": "maravilhoso de ver"
      }
    },
    {
      "latin": "modus operandi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "method of working",
        "fra_Latn": "mode de fonctionnement",
        "spa_Latn": "modo de operar",
        "deu_Latn": "arbeitsweise",
        "ita_Latn": "modo di operare",
        "por_Latn": "modo de operar"
      }
    },
    {
      "latin": "modus vivendi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "way of living",
        "fra_Latn": "mode de vie",
        "spa_Latn": "modo de vivir",
        "deu_Latn": "lebensweise",
        "ita_Latn": "modo di vivere",
        "por_Latn": "modo de viver"
      }
    },
    {
      "latin": "mortui vivos docent",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the dead teach the living",
        "fra_Latn": "les morts enseignent les vivants",
        "spa_Latn": "los muertos ensenan a los vivos",
        "deu_Latn": "die toten lehren die lebenden",
        "ita_Latn": "i morti insegnano ai vivi",
        "por_Latn": "os mortos ensinam os vivos"
      }
    },
    {
      "latin": "mutatis mutandis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "with the necessary changes having been made",
        "fra_Latn": "avec les changements necessaires",
        "spa_Latn": "con los cambios necesarios",
        "deu_Latn": "mit den notigen anderungen",
        "ita_Latn": "con i necessari cambiamenti",
        "por_Latn": "com as devidas mudancas"
      }
    },
    {
      "latin": "naturalia non sunt turpia",
      "category": "phrase",
      "translations": {
        "eng_Latn": "natural things are not shameful",
        "fra_Latn": "les choses naturelles ne sont pas honteuses",
        "spa_Latn": "las cosas naturales no son vergonzosas",
        "deu_Latn": "naturliche dinge sind nicht schandlich",
        "ita_Latn": "le cose naturali non sono vergognose",
        "por_Latn": "as coisas naturais nao sao vergonhosas"
      }
    },
    {
      "latin": "natura nihil frustra facit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "nature does nothing in vain",
        "fra_Latn": "la nature ne fait rien en vain",
        "spa_Latn": "la naturaleza no hace nada en vano",
        "deu_Latn": "die natur tut nichts vergebens",
        "ita_Latn": "la natura non fa nulla invano",
        "por_Latn": "a natureza nada faz em vao"
      }
    },
    {
      "latin": "necessitas non habet legem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "necessity has no law",
        "fra_Latn": "la necessite n a pas de loi",
        "spa_Latn": "la necesidad no tiene ley",
        "deu_Latn": "not kennt kein gebot",
        "ita_Latn": "la necessita non ha legge",
        "por_Latn": "a necessidade nao tem lei"
      }
    },
    {
      "latin": "nemine contradicente",
      "category": "phrase",
      "translations": {
        "eng_Latn": "with no one speaking against",
        "fra_Latn": "sans opposition",
        "spa_Latn": "sin oposicion",
        "deu_Latn": "ohne widerspruch",
        "ita_Latn": "senza opposizione",
        "por_Latn": "sem contradicao"
      }
    },
    {
      "latin": "nihil ad rem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "nothing to the point",
        "fra_Latn": "rien de pertinent",
        "spa_Latn": "nada al respecto",
        "deu_Latn": "nicht zur sache",
        "ita_Latn": "niente in merito",
        "por_Latn": "nada a respeito"
      }
    },
    {
      "latin": "nihil difficile amanti",
      "category": "phrase",
      "translations": {
        "eng_Latn": "nothing is difficult for the one who loves",
        "fra_Latn": "rien n est difficile a celui qui aime",
        "spa_Latn": "nada es dificil para quien ama",
        "deu_Latn": "nichts ist schwer fur den liebenden",
        "ita_Latn": "nulla e difficile per chi ama",
        "por_Latn": "nada e dificil para quem ama"
      }
    },
    {
      "latin": "nihil nisi bonum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "nothing but good",
        "fra_Latn": "rien que du bien",
        "spa_Latn": "nada mas que bien",
        "deu_Latn": "nichts als gutes",
        "ita_Latn": "nulla se non il bene",
        "por_Latn": "nada alem do bem"
      }
    },
    {
      "latin": "nihil novi sub sole",
      "category": "phrase",
      "translations": {
        "eng_Latn": "nothing new under the sun",
        "fra_Latn": "rien de nouveau sous le soleil",
        "spa_Latn": "nada nuevo bajo el sol",
        "deu_Latn": "nichts neues unter der sonne",
        "ita_Latn": "nulla di nuovo sotto il sole",
        "por_Latn": "nada de novo sob o sol"
      }
    },
    {
      "latin": "nil desperandum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "never despair",
        "fra_Latn": "ne desespere jamais",
        "spa_Latn": "nunca desesperes",
        "deu_Latn": "niemals verzweifeln",
        "ita_Latn": "mai disperare",
        "por_Latn": "nunca desesperar"
      }
    },
    {
      "latin": "nisi dominus frustra",
      "category": "phrase",
      "translations": {
        "eng_Latn": "without the lord all is in vain",
        "fra_Latn": "sans le seigneur tout est en vain",
        "spa_Latn": "sin el senor todo es en vano",
        "deu_Latn": "ohne den herrn ist alles vergebens",
        "ita_Latn": "senza il signore tutto e vano",
        "por_Latn": "sem o senhor tudo e em vao"
      }
    },
    {
      "latin": "nolens volens",
      "category": "phrase",
      "translations": {
        "eng_Latn": "willing or unwilling",
        "fra_Latn": "volontairement ou non",
        "spa_Latn": "quiera o no",
        "deu_Latn": "ob man will oder nicht",
        "ita_Latn": "volente o nolente",
        "por_Latn": "quer queira quer nao"
      }
    },
    {
      "latin": "noli me tangere",
      "category": "phrase",
      "translations": {
        "eng_Latn": "do not touch me",
        "fra_Latn": "ne me touche pas",
        "spa_Latn": "no me toques",
        "deu_Latn": "ruh mich nicht an",
        "ita_Latn": "non toccarmi",
        "por_Latn": "nao me toques"
      }
    },
    {
      "latin": "nomen est omen",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the name is a sign",
        "fra_Latn": "le nom est un signe",
        "spa_Latn": "el nombre es un presagio",
        "deu_Latn": "der name ist ein omen",
        "ita_Latn": "il nome e un segno",
        "por_Latn": "o nome e um sinal"
      }
    },
    {
      "latin": "non ducor duco",
      "category": "phrase",
      "translations": {
        "eng_Latn": "i am not led i lead",
        "fra_Latn": "je ne suis pas conduit je conduis",
        "spa_Latn": "no soy guiado guio",
        "deu_Latn": "ich werde nicht gefuhrt ich fuhre",
        "ita_Latn": "non sono condotto conduco",
        "por_Latn": "nao sou conduzido conduzo"
      }
    },
    {
      "latin": "non enim ad astra mollis e terris via",
      "category": "phrase",
      "translations": {
        "eng_Latn": "there is no easy way from the earth to the stars",
        "fra_Latn": "il n y a pas de voie facile de la terre aux etoiles",
        "spa_Latn": "no hay camino facil de la tierra a las estrellas",
        "deu_Latn": "es gibt keinen leichten weg von der erde zu den sternen",
        "ita_Latn": "non c e via facile dalla terra alle stelle",
        "por_Latn": "nao ha caminho facil da terra as estrelas"
      }
    },
    {
      "latin": "non est ad astra mollis e terris via",
      "category": "phrase",
      "translations": {
        "eng_Latn": "there is no easy way from earth to the stars",
        "fra_Latn": "il n y a pas de voie facile de la terre aux etoiles",
        "spa_Latn": "no hay camino facil de la tierra a las estrellas",
        "deu_Latn": "es gibt keinen leichten weg von der erde zu den sternen",
        "ita_Latn": "non esiste via facile dalla terra alle stelle",
        "por_Latn": "nao existe caminho facil da terra as estrelas"
      }
    },
    {
      "latin": "non nobis solum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "not for ourselves alone",
        "fra_Latn": "pas pour nous seuls",
        "spa_Latn": "no solo para nosotros",
        "deu_Latn": "nicht fur uns allein",
        "ita_Latn": "non solo per noi stessi",
        "por_Latn": "nao apenas para nos"
      }
    },
    {
      "latin": "non omnis moriar",
      "category": "phrase",
      "translations": {
        "eng_Latn": "i shall not wholly die",
        "fra_Latn": "je ne mourrai pas tout entier",
        "spa_Latn": "no morire del todo",
        "deu_Latn": "ich werde nicht ganz sterben",
        "ita_Latn": "non moriro del tutto",
        "por_Latn": "nao morrerei por inteiro"
      }
    },
    {
      "latin": "non plus ultra",
      "category": "phrase",
      "translations": {
        "eng_Latn": "nothing further beyond",
        "fra_Latn": "rien au dela",
        "spa_Latn": "nada mas alla",
        "deu_Latn": "nichts weiter daruber hinaus",
        "ita_Latn": "niente oltre",
        "por_Latn": "nada mais alem"
      }
    },
    {
      "latin": "non scholae sed vitae discimus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "we learn not for school but for life",
        "fra_Latn": "nous apprenons non pour l ecole mais pour la vie",
        "spa_Latn": "aprendemos no para la escuela sino para la vida",
        "deu_Latn": "wir lernen nicht fur die schule sondern fur das leben",
        "ita_Latn": "impariamo non per la scuola ma per la vita",
        "por_Latn": "aprendemos nao para a escola mas para a vida"
      }
    },
    {
      "latin": "nosce te ipsum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "know yourself",
        "fra_Latn": "connais toi toi meme",
        "spa_Latn": "conocete a ti mismo",
        "deu_Latn": "erkenne dich selbst",
        "ita_Latn": "conosci te stesso",
        "por_Latn": "conhece te a ti mesmo"
      }
    },
    {
      "latin": "nova et vetera",
      "category": "phrase",
      "translations": {
        "eng_Latn": "new and old things",
        "fra_Latn": "choses nouvelles et anciennes",
        "spa_Latn": "cosas nuevas y viejas",
        "deu_Latn": "neues und altes",
        "ita_Latn": "cose nuove e vecchie",
        "por_Latn": "coisas novas e velhas"
      }
    },
    {
      "latin": "nulla dies sine linea",
      "category": "phrase",
      "translations": {
        "eng_Latn": "no day without a line",
        "fra_Latn": "pas un jour sans une ligne",
        "spa_Latn": "ningun dia sin una linea",
        "deu_Latn": "kein tag ohne zeile",
        "ita_Latn": "nessun giorno senza una linea",
        "por_Latn": "nenhum dia sem uma linha"
      }
    },
    {
      "latin": "nulla poena sine lege",
      "category": "phrase",
      "translations": {
        "eng_Latn": "no penalty without law",
        "fra_Latn": "pas de peine sans loi",
        "spa_Latn": "no hay pena sin ley",
        "deu_Latn": "keine strafe ohne gesetz",
        "ita_Latn": "nessuna pena senza legge",
        "por_Latn": "nenhuma pena sem lei"
      }
    },
    {
      "latin": "nulla tenaci invia est via",
      "category": "phrase",
      "translations": {
        "eng_Latn": "for the tenacious no road is impassable",
        "fra_Latn": "pour le tenace aucune route n est infranchissable",
        "spa_Latn": "para el tenaz ningun camino es intransitable",
        "deu_Latn": "fur den beharrlichen ist kein weg ungangbar",
        "ita_Latn": "per il tenace nessuna via e impraticabile",
        "por_Latn": "para o tenaz nenhum caminho e intransponivel"
      }
    },
    {
      "latin": "numquam retrorsum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "never backward",
        "fra_Latn": "jamais en arriere",
        "spa_Latn": "nunca hacia atras",
        "deu_Latn": "niemals ruckwarts",
        "ita_Latn": "mai indietro",
        "por_Latn": "nunca para tras"
      }
    },
    {
      "latin": "numquam non paratus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "never unprepared",
        "fra_Latn": "jamais sans preparation",
        "spa_Latn": "nunca desprevenido",
        "deu_Latn": "niemals unvorbereitet",
        "ita_Latn": "mai impreparato",
        "por_Latn": "nunca despreparado"
      }
    },
    {
      "latin": "omne bonum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "every good thing",
        "fra_Latn": "tout bien",
        "spa_Latn": "todo bien",
        "deu_Latn": "alles gute",
        "ita_Latn": "ogni bene",
        "por_Latn": "todo bem"
      }
    },
    {
      "latin": "omne ignotum pro magnifico",
      "category": "phrase",
      "translations": {
        "eng_Latn": "everything unknown is taken for magnificent",
        "fra_Latn": "tout ce qui est inconnu est tenu pour magnifique",
        "spa_Latn": "todo lo desconocido se toma por magnifico",
        "deu_Latn": "alles unbekannte gilt als grossartig",
        "ita_Latn": "tutto cio che e ignoto e considerato magnifico",
        "por_Latn": "tudo o que e desconhecido e tomado por magnifico"
      }
    },
    {
      "latin": "omne initium difficile est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "every beginning is difficult",
        "fra_Latn": "tout commencement est difficile",
        "spa_Latn": "todo comienzo es dificil",
        "deu_Latn": "jeder anfang ist schwer",
        "ita_Latn": "ogni inizio e difficile",
        "por_Latn": "todo comeco e dificil"
      }
    },
    {
      "latin": "omne trium perfectum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "everything that comes in threes is perfect",
        "fra_Latn": "tout ce qui vient par trois est parfait",
        "spa_Latn": "todo lo que viene en tres es perfecto",
        "deu_Latn": "alles was in dreien kommt ist vollkommen",
        "ita_Latn": "tutto cio che viene in tre e perfetto",
        "por_Latn": "tudo o que vem em tres e perfeito"
      }
    },
    {
      "latin": "omnia dicta fortiora si dicta latina",
      "category": "phrase",
      "translations": {
        "eng_Latn": "everything said is stronger if said in latin",
        "fra_Latn": "tout ce qui est dit est plus fort si c est dit en latin",
        "spa_Latn": "todo lo dicho es mas fuerte si se dice en latin",
        "deu_Latn": "alles gesagte ist starker wenn es auf latein gesagt wird",
        "ita_Latn": "tutto cio che e detto e piu forte se detto in latino",
        "por_Latn": "tudo o que e dito fica mais forte se dito em latim"
      }
    },
    {
      "latin": "omnia mea mecum porto",
      "category": "phrase",
      "translations": {
        "eng_Latn": "all that is mine i carry with me",
        "fra_Latn": "je porte avec moi tout ce qui est a moi",
        "spa_Latn": "llevo conmigo todo lo que es mio",
        "deu_Latn": "alles meine trage ich mit mir",
        "ita_Latn": "porto con me tutto cio che e mio",
        "por_Latn": "trago comigo tudo o que e meu"
      }
    },
    {
      "latin": "omnia mutantur nos et mutamur in illis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "all things change and we change with them",
        "fra_Latn": "toutes choses changent et nous changeons avec elles",
        "spa_Latn": "todas las cosas cambian y nosotros cambiamos con ellas",
        "deu_Latn": "alles verandert sich und wir uns mit ihnen",
        "ita_Latn": "tutte le cose cambiano e noi cambiamo con esse",
        "por_Latn": "todas as coisas mudam e nos mudamos com elas"
      }
    },
    {
      "latin": "omnia praeclara rara",
      "category": "phrase",
      "translations": {
        "eng_Latn": "all excellent things are rare",
        "fra_Latn": "toutes les choses excellentes sont rares",
        "spa_Latn": "todas las cosas excelentes son raras",
        "deu_Latn": "alle hervorragenden dinge sind selten",
        "ita_Latn": "tutte le cose eccellenti sono rare",
        "por_Latn": "todas as coisas excelentes sao raras"
      }
    },
    {
      "latin": "omnia vincit labor",
      "category": "phrase",
      "translations": {
        "eng_Latn": "work conquers all things",
        "fra_Latn": "le travail vainc tout",
        "spa_Latn": "el trabajo vence todo",
        "deu_Latn": "arbeit besiegt alles",
        "ita_Latn": "il lavoro vince tutto",
        "por_Latn": "o trabalho vence tudo"
      }
    },
    {
      "latin": "optimum est pati quod emendare non possis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "it is best to endure what you cannot remedy",
        "fra_Latn": "il est meilleur de supporter ce que l on ne peut corriger",
        "spa_Latn": "es mejor soportar lo que no puedes corregir",
        "deu_Latn": "es ist am besten zu ertragen was du nicht andern kannst",
        "ita_Latn": "e meglio sopportare cio che non puoi correggere",
        "por_Latn": "e melhor suportar o que nao podes corrigir"
      }
    },
    {
      "latin": "ora et labora",
      "category": "phrase",
      "translations": {
        "eng_Latn": "pray and work",
        "fra_Latn": "prie et travaille",
        "spa_Latn": "ora y trabaja",
        "deu_Latn": "bete und arbeite",
        "ita_Latn": "prega e lavora",
        "por_Latn": "reza e trabalha"
      }
    },
    {
      "latin": "ordo ab chao",
      "category": "phrase",
      "translations": {
        "eng_Latn": "order out of chaos",
        "fra_Latn": "ordre issu du chaos",
        "spa_Latn": "orden del caos",
        "deu_Latn": "ordnung aus dem chaos",
        "ita_Latn": "ordine dal caos",
        "por_Latn": "ordem a partir do caos"
      }
    },
    {
      "latin": "os homini sublime dedit caelum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "heaven gave man an uplifted face",
        "fra_Latn": "le ciel a donne a l homme un visage tourne vers le haut",
        "spa_Latn": "el cielo dio al hombre un rostro erguido",
        "deu_Latn": "der himmel gab dem menschen ein erhobenes gesicht",
        "ita_Latn": "il cielo diede all uomo un volto rivolto in alto",
        "por_Latn": "o ceu deu ao homem um rosto erguido"
      }
    },
    {
      "latin": "pacta sunt servanda",
      "category": "phrase",
      "translations": {
        "eng_Latn": "agreements must be kept",
        "fra_Latn": "les accords doivent etre respectes",
        "spa_Latn": "los pactos deben cumplirse",
        "deu_Latn": "vertrage sind einzuhalten",
        "ita_Latn": "i patti devono essere rispettati",
        "por_Latn": "os pactos devem ser cumpridos"
      }
    },
    {
      "latin": "panem et circenses",
      "category": "phrase",
      "translations": {
        "eng_Latn": "bread and circuses",
        "fra_Latn": "du pain et des jeux",
        "spa_Latn": "pan y circo",
        "deu_Latn": "brot und spiele",
        "ita_Latn": "pane e giochi",
        "por_Latn": "pao e circo"
      }
    },
    {
      "latin": "par pari refertur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "like is returned for like",
        "fra_Latn": "le semblable est rendu pour le semblable",
        "spa_Latn": "lo semejante se devuelve por lo semejante",
        "deu_Latn": "gleiches wird mit gleichem vergolten",
        "ita_Latn": "il simile e reso con il simile",
        "por_Latn": "o semelhante e retribuido com o semelhante"
      }
    },
    {
      "latin": "parce sepulto",
      "category": "phrase",
      "translations": {
        "eng_Latn": "spare the buried",
        "fra_Latn": "epargne le tombe",
        "spa_Latn": "respeta al sepultado",
        "deu_Latn": "schone den begrabenen",
        "ita_Latn": "risparmia il sepolto",
        "por_Latn": "poupa o sepultado"
      }
    },
    {
      "latin": "parva sed apta",
      "category": "phrase",
      "translations": {
        "eng_Latn": "small but suitable",
        "fra_Latn": "petit mais adapte",
        "spa_Latn": "pequeno pero adecuado",
        "deu_Latn": "klein aber passend",
        "ita_Latn": "piccolo ma adatto",
        "por_Latn": "pequeno mas adequado"
      }
    },
    {
      "latin": "pater familias",
      "category": "phrase",
      "translations": {
        "eng_Latn": "head of the family",
        "fra_Latn": "chef de famille",
        "spa_Latn": "padre de familia",
        "deu_Latn": "familienoberhaupt",
        "ita_Latn": "capofamiglia",
        "por_Latn": "chefe da familia"
      }
    },
    {
      "latin": "pax aeterna",
      "category": "phrase",
      "translations": {
        "eng_Latn": "eternal peace",
        "fra_Latn": "paix eternelle",
        "spa_Latn": "paz eterna",
        "deu_Latn": "ewiger friede",
        "ita_Latn": "pace eterna",
        "por_Latn": "paz eterna"
      }
    },
    {
      "latin": "pax deorum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "peace of the gods",
        "fra_Latn": "paix des dieux",
        "spa_Latn": "paz de los dioses",
        "deu_Latn": "friede der gotter",
        "ita_Latn": "pace degli dei",
        "por_Latn": "paz dos deuses"
      }
    },
    {
      "latin": "pax et bonum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "peace and good",
        "fra_Latn": "paix et bien",
        "spa_Latn": "paz y bien",
        "deu_Latn": "frieden und gutes",
        "ita_Latn": "pace e bene",
        "por_Latn": "paz e bem"
      }
    },
    {
      "latin": "pax in terra",
      "category": "phrase",
      "translations": {
        "eng_Latn": "peace on earth",
        "fra_Latn": "paix sur terre",
        "spa_Latn": "paz en la tierra",
        "deu_Latn": "friede auf erden",
        "ita_Latn": "pace in terra",
        "por_Latn": "paz na terra"
      }
    },
    {
      "latin": "per capita",
      "category": "phrase",
      "translations": {
        "eng_Latn": "per head",
        "fra_Latn": "par tete",
        "spa_Latn": "por cabeza",
        "deu_Latn": "pro kopf",
        "ita_Latn": "pro capite",
        "por_Latn": "por cabeca"
      }
    },
    {
      "latin": "per contra",
      "category": "phrase",
      "translations": {
        "eng_Latn": "on the contrary",
        "fra_Latn": "au contraire",
        "spa_Latn": "por el contrario",
        "deu_Latn": "im gegenteil",
        "ita_Latn": "al contrario",
        "por_Latn": "ao contrario"
      }
    },
    {
      "latin": "per curiam",
      "category": "phrase",
      "translations": {
        "eng_Latn": "by the court",
        "fra_Latn": "par la cour",
        "spa_Latn": "por el tribunal",
        "deu_Latn": "durch das gericht",
        "ita_Latn": "per la corte",
        "por_Latn": "pelo tribunal"
      }
    },
    {
      "latin": "per diem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "per day",
        "fra_Latn": "par jour",
        "spa_Latn": "por dia",
        "deu_Latn": "pro tag",
        "ita_Latn": "al giorno",
        "por_Latn": "por dia"
      }
    },
    {
      "latin": "per mare per terram",
      "category": "phrase",
      "translations": {
        "eng_Latn": "by sea and by land",
        "fra_Latn": "par mer et par terre",
        "spa_Latn": "por mar y por tierra",
        "deu_Latn": "zu wasser und zu land",
        "ita_Latn": "per mare e per terra",
        "por_Latn": "por mar e por terra"
      }
    },
    {
      "latin": "per se",
      "category": "phrase",
      "translations": {
        "eng_Latn": "by itself",
        "fra_Latn": "en soi",
        "spa_Latn": "por si mismo",
        "deu_Latn": "an sich",
        "ita_Latn": "di per se",
        "por_Latn": "por si so"
      }
    },
    {
      "latin": "persona non grata",
      "category": "phrase",
      "translations": {
        "eng_Latn": "unwelcome person",
        "fra_Latn": "personne non grata",
        "spa_Latn": "persona no grata",
        "deu_Latn": "unerwunschte person",
        "ita_Latn": "persona non gradita",
        "por_Latn": "pessoa nao grata"
      }
    },
    {
      "latin": "petitio principii",
      "category": "phrase",
      "translations": {
        "eng_Latn": "begging the question",
        "fra_Latn": "petition de principe",
        "spa_Latn": "peticion de principio",
        "deu_Latn": "zirkelschluss",
        "ita_Latn": "petizione di principio",
        "por_Latn": "peticao de principio"
      }
    },
    {
      "latin": "plus ultra",
      "category": "phrase",
      "translations": {
        "eng_Latn": "further beyond",
        "fra_Latn": "plus loin encore",
        "spa_Latn": "mas alla",
        "deu_Latn": "immer weiter",
        "ita_Latn": "piu oltre",
        "por_Latn": "mais alem"
      }
    },
    {
      "latin": "post factum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "after the fact",
        "fra_Latn": "apres le fait",
        "spa_Latn": "despues del hecho",
        "deu_Latn": "nach der tat",
        "ita_Latn": "dopo il fatto",
        "por_Latn": "apos o fato"
      }
    },
    {
      "latin": "post meridiem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "after midday",
        "fra_Latn": "apres midi",
        "spa_Latn": "despues del mediodia",
        "deu_Latn": "nachmittag",
        "ita_Latn": "dopo mezzogiorno",
        "por_Latn": "depois do meio dia"
      }
    },
    {
      "latin": "post mortem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "after death",
        "fra_Latn": "apres la mort",
        "spa_Latn": "despues de la muerte",
        "deu_Latn": "nach dem tod",
        "ita_Latn": "dopo la morte",
        "por_Latn": "apos a morte"
      }
    },
    {
      "latin": "post scriptum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "written after",
        "fra_Latn": "ecrit apres",
        "spa_Latn": "escrito despues",
        "deu_Latn": "nachschrift",
        "ita_Latn": "scritto dopo",
        "por_Latn": "escrito depois"
      }
    },
    {
      "latin": "praemonitus praemunitus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "forewarned is forearmed",
        "fra_Latn": "averti a l avance c est etre arme d avance",
        "spa_Latn": "advertido es prevenido",
        "deu_Latn": "vorgewarnt ist gewappnet",
        "ita_Latn": "uomo avvisato mezzo salvato",
        "por_Latn": "avisado esta prevenido"
      }
    },
    {
      "latin": "praesidium et dulce decus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "protection and sweet glory",
        "fra_Latn": "protection et douce gloire",
        "spa_Latn": "proteccion y dulce gloria",
        "deu_Latn": "schutz und suser ruhm",
        "ita_Latn": "protezione e dolce gloria",
        "por_Latn": "protecao e doce gloria"
      }
    },
    {
      "latin": "primum non nocere",
      "category": "phrase",
      "translations": {
        "eng_Latn": "first do no harm",
        "fra_Latn": "d abord ne pas nuire",
        "spa_Latn": "primero no hacer dano",
        "deu_Latn": "zuerst nicht schaden",
        "ita_Latn": "per prima cosa non nuocere",
        "por_Latn": "primeiro nao causar dano"
      }
    },
    {
      "latin": "principia probant non probantur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "first principles prove they are not proved",
        "fra_Latn": "les premiers principes prouvent mais ne sont pas prouves",
        "spa_Latn": "los primeros principios prueban pero no son probados",
        "deu_Latn": "erste prinzipien beweisen und werden nicht bewiesen",
        "ita_Latn": "i primi principi provano ma non sono provati",
        "por_Latn": "os primeiros principios provam e nao sao provados"
      }
    },
    {
      "latin": "pro aris et focis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "for altars and hearths",
        "fra_Latn": "pour les autels et les foyers",
        "spa_Latn": "por los altares y los hogares",
        "deu_Latn": "fur altar und herd",
        "ita_Latn": "per altari e focolari",
        "por_Latn": "pelos altares e lares"
      }
    },
    {
      "latin": "pro bono",
      "category": "phrase",
      "translations": {
        "eng_Latn": "for the public good",
        "fra_Latn": "pour le bien public",
        "spa_Latn": "por el bien publico",
        "deu_Latn": "fur das gemeinwohl",
        "ita_Latn": "per il bene pubblico",
        "por_Latn": "para o bem publico"
      }
    },
    {
      "latin": "pro forma",
      "category": "phrase",
      "translations": {
        "eng_Latn": "for form s sake",
        "fra_Latn": "pour la forme",
        "spa_Latn": "por formalidad",
        "deu_Latn": "pro forma",
        "ita_Latn": "per forma",
        "por_Latn": "pro forma"
      }
    },
    {
      "latin": "pro patria",
      "category": "phrase",
      "translations": {
        "eng_Latn": "for country",
        "fra_Latn": "pour la patrie",
        "spa_Latn": "por la patria",
        "deu_Latn": "fur das vaterland",
        "ita_Latn": "per la patria",
        "por_Latn": "pela patria"
      }
    },
    {
      "latin": "pro rata",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in proportion",
        "fra_Latn": "au prorata",
        "spa_Latn": "a prorrata",
        "deu_Latn": "anteilig",
        "ita_Latn": "pro rata",
        "por_Latn": "proporcionalmente"
      }
    },
    {
      "latin": "pro re nata",
      "category": "phrase",
      "translations": {
        "eng_Latn": "as circumstances arise",
        "fra_Latn": "selon les circonstances",
        "spa_Latn": "segun surja la circunstancia",
        "deu_Latn": "je nach lage",
        "ita_Latn": "secondo necessita",
        "por_Latn": "conforme a necessidade"
      }
    },
    {
      "latin": "pro tempore",
      "category": "phrase",
      "translations": {
        "eng_Latn": "for the time being",
        "fra_Latn": "pour le moment",
        "spa_Latn": "por el momento",
        "deu_Latn": "vorlaufig",
        "ita_Latn": "per il momento",
        "por_Latn": "por enquanto"
      }
    },
    {
      "latin": "probatum est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "it has been tested",
        "fra_Latn": "cela a ete eprouve",
        "spa_Latn": "ha sido probado",
        "deu_Latn": "es ist erprobt",
        "ita_Latn": "e stato provato",
        "por_Latn": "foi testado"
      }
    },
    {
      "latin": "quae nocent docent",
      "category": "phrase",
      "translations": {
        "eng_Latn": "things that harm teach",
        "fra_Latn": "les choses qui blessent enseignent",
        "spa_Latn": "las cosas que hacen dano ensenan",
        "deu_Latn": "was schadet lehrt",
        "ita_Latn": "le cose che nuocciono insegnano",
        "por_Latn": "as coisas que ferem ensinam"
      }
    },
    {
      "latin": "quaere verum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "seek the truth",
        "fra_Latn": "cherche la verite",
        "spa_Latn": "busca la verdad",
        "deu_Latn": "suche die wahrheit",
        "ita_Latn": "cerca la verita",
        "por_Latn": "procura a verdade"
      }
    },
    {
      "latin": "quaerite primum regnum dei",
      "category": "phrase",
      "translations": {
        "eng_Latn": "seek first the kingdom of god",
        "fra_Latn": "cherchez d abord le royaume de dieu",
        "spa_Latn": "buscad primero el reino de dios",
        "deu_Latn": "suchet zuerst das reich gottes",
        "ita_Latn": "cercate prima il regno di dio",
        "por_Latn": "buscai primeiro o reino de deus"
      }
    },
    {
      "latin": "quare fremuerunt gentes",
      "category": "phrase",
      "translations": {
        "eng_Latn": "why have the nations raged",
        "fra_Latn": "pourquoi les nations se sont elles agitees",
        "spa_Latn": "por que se enfurecieron las naciones",
        "deu_Latn": "warum toben die volker",
        "ita_Latn": "perche fremono le genti",
        "por_Latn": "por que se enfureceram as nacoes"
      }
    },
    {
      "latin": "qui audet adipiscitur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "who dares wins",
        "fra_Latn": "qui ose gagne",
        "spa_Latn": "quien se atreve gana",
        "deu_Latn": "wer wagt gewinnt",
        "ita_Latn": "chi osa vince",
        "por_Latn": "quem ousa vence"
      }
    },
    {
      "latin": "qui bene amat bene castigat",
      "category": "phrase",
      "translations": {
        "eng_Latn": "he who loves well chastises well",
        "fra_Latn": "qui aime bien corrige bien",
        "spa_Latn": "quien bien ama bien castiga",
        "deu_Latn": "wer gut liebt zuchtigt gut",
        "ita_Latn": "chi ama bene castiga bene",
        "por_Latn": "quem ama bem castiga bem"
      }
    },
    {
      "latin": "qui desiderat pacem praeparet bellum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "let him who desires peace prepare for war",
        "fra_Latn": "que celui qui desire la paix prepare la guerre",
        "spa_Latn": "quien desee la paz prepare la guerra",
        "deu_Latn": "wer frieden will bereite den krieg vor",
        "ita_Latn": "chi desidera la pace prepari la guerra",
        "por_Latn": "quem deseja a paz prepare a guerra"
      }
    },
    {
      "latin": "qui non laborat non manducet",
      "category": "phrase",
      "translations": {
        "eng_Latn": "he who does not work shall not eat",
        "fra_Latn": "qui ne travaille pas qu il ne mange pas",
        "spa_Latn": "quien no trabaja que no coma",
        "deu_Latn": "wer nicht arbeitet soll auch nicht essen",
        "ita_Latn": "chi non lavora non mangi",
        "por_Latn": "quem nao trabalha nao coma"
      }
    },
    {
      "latin": "qui quaerit invenit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "he who seeks finds",
        "fra_Latn": "qui cherche trouve",
        "spa_Latn": "quien busca encuentra",
        "deu_Latn": "wer sucht der findet",
        "ita_Latn": "chi cerca trova",
        "por_Latn": "quem procura encontra"
      }
    },
    {
      "latin": "qui scribit bis legit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "he who writes reads twice",
        "fra_Latn": "qui ecrit lit deux fois",
        "spa_Latn": "quien escribe lee dos veces",
        "deu_Latn": "wer schreibt liest zweimal",
        "ita_Latn": "chi scrive legge due volte",
        "por_Latn": "quem escreve le duas vezes"
      }
    },
    {
      "latin": "quia suam uxorem etiam suspicione vacare vellet",
      "category": "phrase",
      "translations": {
        "eng_Latn": "because he wished his wife to be free even from suspicion",
        "fra_Latn": "parce qu il voulait que son epouse soit exempte meme de soupcon",
        "spa_Latn": "porque queria que su esposa estuviera libre incluso de sospecha",
        "deu_Latn": "weil er wollte dass seine frau sogar frei von verdacht sei",
        "ita_Latn": "perche voleva che sua moglie fosse libera anche dal sospetto",
        "por_Latn": "porque desejava que sua esposa estivesse livre ate de suspeita"
      }
    },
    {
      "latin": "quid agis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "how are you doing",
        "fra_Latn": "comment vas tu",
        "spa_Latn": "como estas",
        "deu_Latn": "wie geht es dir",
        "ita_Latn": "come stai",
        "por_Latn": "como vais"
      }
    },
    {
      "latin": "quid est veritas",
      "category": "phrase",
      "translations": {
        "eng_Latn": "what is truth",
        "fra_Latn": "qu est ce que la verite",
        "spa_Latn": "que es la verdad",
        "deu_Latn": "was ist wahrheit",
        "ita_Latn": "che cos e la verita",
        "por_Latn": "o que e a verdade"
      }
    },
    {
      "latin": "quid nunc",
      "category": "phrase",
      "translations": {
        "eng_Latn": "what now",
        "fra_Latn": "quoi maintenant",
        "spa_Latn": "que ahora",
        "deu_Latn": "was nun",
        "ita_Latn": "e ora",
        "por_Latn": "e agora"
      }
    },
    {
      "latin": "quid pro quo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "something for something",
        "fra_Latn": "quelque chose pour quelque chose",
        "spa_Latn": "algo por algo",
        "deu_Latn": "etwas fur etwas",
        "ita_Latn": "qualcosa per qualcosa",
        "por_Latn": "algo por algo"
      }
    },
    {
      "latin": "quies",
      "category": "word",
      "translations": {
        "eng_Latn": "rest",
        "fra_Latn": "repos",
        "spa_Latn": "reposo",
        "deu_Latn": "ruhe",
        "ita_Latn": "riposo",
        "por_Latn": "descanso"
      }
    },
    {
      "latin": "quis custodiet ipsos custodes",
      "category": "phrase",
      "translations": {
        "eng_Latn": "who will guard the guards themselves",
        "fra_Latn": "qui gardera les gardiens eux memes",
        "spa_Latn": "quien vigilara a los propios vigilantes",
        "deu_Latn": "wer bewacht die wachter selbst",
        "ita_Latn": "chi sorvegliera i sorveglianti stessi",
        "por_Latn": "quem guardara os proprios guardioes"
      }
    },
    {
      "latin": "quo vadis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "where are you going",
        "fra_Latn": "ou vas tu",
        "spa_Latn": "adonde vas",
        "deu_Latn": "wohin gehst du",
        "ita_Latn": "dove vai",
        "por_Latn": "aonde vais"
      }
    },
    {
      "latin": "quod erat demonstrandum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "which was to be demonstrated",
        "fra_Latn": "ce qu il fallait demontrer",
        "spa_Latn": "lo que habia que demostrar",
        "deu_Latn": "was zu beweisen war",
        "ita_Latn": "come volevasi dimostrare",
        "por_Latn": "o que se queria demonstrar"
      }
    },
    {
      "latin": "quod gratis asseritur gratis negatur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "what is asserted without proof may be denied without proof",
        "fra_Latn": "ce qui est affirme sans preuve peut etre nie sans preuve",
        "spa_Latn": "lo que se afirma sin prueba puede negarse sin prueba",
        "deu_Latn": "was ohne beweis behauptet wird kann ohne beweis bestritten werden",
        "ita_Latn": "cio che e affermato senza prova puo essere negato senza prova",
        "por_Latn": "o que e afirmado sem prova pode ser negado sem prova"
      }
    },
    {
      "latin": "quod licet iovi non licet bovi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "what is permitted to jupiter is not permitted to the ox",
        "fra_Latn": "ce qui est permis a jupiter ne l est pas au boeuf",
        "spa_Latn": "lo que le es licito a jupiter no le es licito al buey",
        "deu_Latn": "was jupiter erlaubt ist ist dem ochsen nicht erlaubt",
        "ita_Latn": "cio che e lecito a giove non e lecito al bue",
        "por_Latn": "o que e permitido a jupiter nao e permitido ao boi"
      }
    },
    {
      "latin": "quod me nutrit me destruit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "what nourishes me destroys me",
        "fra_Latn": "ce qui me nourrit me detruit",
        "spa_Latn": "lo que me nutre me destruye",
        "deu_Latn": "was mich nahrt zerstort mich",
        "ita_Latn": "cio che mi nutre mi distrugge",
        "por_Latn": "o que me nutre me destroi"
      }
    },
    {
      "latin": "quod scripsi scripsi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "what i have written i have written",
        "fra_Latn": "ce que j ai ecrit je l ai ecrit",
        "spa_Latn": "lo que he escrito he escrito",
        "deu_Latn": "was ich geschrieben habe habe ich geschrieben",
        "ita_Latn": "cio che ho scritto ho scritto",
        "por_Latn": "o que escrevi escrevi"
      }
    },
    {
      "latin": "quondam",
      "category": "word",
      "translations": {
        "eng_Latn": "formerly",
        "fra_Latn": "autrefois",
        "spa_Latn": "antiguamente",
        "deu_Latn": "ehemals",
        "ita_Latn": "un tempo",
        "por_Latn": "antigamente"
      }
    },
    {
      "latin": "rebus sic stantibus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "with things standing thus",
        "fra_Latn": "les choses etant ainsi",
        "spa_Latn": "estando asi las cosas",
        "deu_Latn": "unter diesen umstanden",
        "ita_Latn": "stando cosi le cose",
        "por_Latn": "estando assim as coisas"
      }
    },
    {
      "latin": "recte et fideliter",
      "category": "phrase",
      "translations": {
        "eng_Latn": "uprightly and faithfully",
        "fra_Latn": "droiture et fidelite",
        "spa_Latn": "rectamente y fielmente",
        "deu_Latn": "rechtschaffen und treu",
        "ita_Latn": "rettamente e fedelmente",
        "por_Latn": "retamente e fielmente"
      }
    },
    {
      "latin": "regnat populus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the people rule",
        "fra_Latn": "le peuple regne",
        "spa_Latn": "el pueblo gobierna",
        "deu_Latn": "das volk herrscht",
        "ita_Latn": "il popolo regna",
        "por_Latn": "o povo governa"
      }
    },
    {
      "latin": "requiescat in pace",
      "category": "phrase",
      "translations": {
        "eng_Latn": "may he or she rest in peace",
        "fra_Latn": "qu il ou elle repose en paix",
        "spa_Latn": "descanse en paz",
        "deu_Latn": "ruhe in frieden",
        "ita_Latn": "riposi in pace",
        "por_Latn": "descanse em paz"
      }
    },
    {
      "latin": "res ipsa loquitur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the thing speaks for itself",
        "fra_Latn": "la chose parle d elle meme",
        "spa_Latn": "la cosa habla por si sola",
        "deu_Latn": "die sache spricht fur sich selbst",
        "ita_Latn": "la cosa parla da se",
        "por_Latn": "a coisa fala por si"
      }
    },
    {
      "latin": "res judicata",
      "category": "phrase",
      "translations": {
        "eng_Latn": "a matter judged",
        "fra_Latn": "chose jugee",
        "spa_Latn": "cosa juzgada",
        "deu_Latn": "rechtskraftige sache",
        "ita_Latn": "cosa giudicata",
        "por_Latn": "coisa julgada"
      }
    },
    {
      "latin": "res non verba",
      "category": "phrase",
      "translations": {
        "eng_Latn": "deeds not words",
        "fra_Latn": "des actes pas des paroles",
        "spa_Latn": "hechos no palabras",
        "deu_Latn": "taten nicht worte",
        "ita_Latn": "fatti non parole",
        "por_Latn": "atos nao palavras"
      }
    },
    {
      "latin": "revera",
      "category": "word",
      "translations": {
        "eng_Latn": "in truth",
        "fra_Latn": "en verite",
        "spa_Latn": "en verdad",
        "deu_Latn": "in wahrheit",
        "ita_Latn": "in verita",
        "por_Latn": "na verdade"
      }
    },
    {
      "latin": "rigor mortis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "stiffness of death",
        "fra_Latn": "rigidite de la mort",
        "spa_Latn": "rigidez de la muerte",
        "deu_Latn": "totenstarre",
        "ita_Latn": "rigidita della morte",
        "por_Latn": "rigidez da morte"
      }
    },
    {
      "latin": "rosa sine spina",
      "category": "phrase",
      "translations": {
        "eng_Latn": "a rose without a thorn",
        "fra_Latn": "une rose sans epine",
        "spa_Latn": "una rosa sin espina",
        "deu_Latn": "eine rose ohne dorn",
        "ita_Latn": "una rosa senza spina",
        "por_Latn": "uma rosa sem espinho"
      }
    },
    {
      "latin": "salus populi suprema lex esto",
      "category": "phrase",
      "translations": {
        "eng_Latn": "let the welfare of the people be the supreme law",
        "fra_Latn": "que le salut du peuple soit la loi supreme",
        "spa_Latn": "que la salud del pueblo sea la ley suprema",
        "deu_Latn": "das wohl des volkes sei oberstes gesetz",
        "ita_Latn": "la salvezza del popolo sia la legge suprema",
        "por_Latn": "o bem estar do povo seja a lei suprema"
      }
    },
    {
      "latin": "sapere aude",
      "category": "phrase",
      "translations": {
        "eng_Latn": "dare to know",
        "fra_Latn": "ose savoir",
        "spa_Latn": "atrevete a saber",
        "deu_Latn": "habe mut dich deines verstandes zu bedienen",
        "ita_Latn": "abbi il coraggio di sapere",
        "por_Latn": "ousa saber"
      }
    },
    {
      "latin": "scientia est lux lucis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "knowledge is the light of light",
        "fra_Latn": "le savoir est la lumiere de la lumiere",
        "spa_Latn": "el conocimiento es la luz de la luz",
        "deu_Latn": "wissen ist das licht des lichts",
        "ita_Latn": "la conoscenza e la luce della luce",
        "por_Latn": "o conhecimento e a luz da luz"
      }
    },
    {
      "latin": "scientia imperii decus et tutamen",
      "category": "phrase",
      "translations": {
        "eng_Latn": "knowledge is the adornment and safeguard of the realm",
        "fra_Latn": "le savoir est l ornement et la sauvegarde du royaume",
        "spa_Latn": "el conocimiento es el adorno y la defensa del reino",
        "deu_Latn": "wissen ist schmuck und schutz des reiches",
        "ita_Latn": "la conoscenza e ornamento e tutela del regno",
        "por_Latn": "o conhecimento e ornamento e defesa do reino"
      }
    },
    {
      "latin": "semel in anno licet insanire",
      "category": "phrase",
      "translations": {
        "eng_Latn": "once a year it is permitted to go mad",
        "fra_Latn": "une fois l an il est permis de devenir fou",
        "spa_Latn": "una vez al ano se permite enloquecer",
        "deu_Latn": "einmal im jahr darf man verruckt sein",
        "ita_Latn": "una volta l anno e lecito impazzire",
        "por_Latn": "uma vez por ano e permitido enlouquecer"
      }
    },
    {
      "latin": "semper ad meliora",
      "category": "phrase",
      "translations": {
        "eng_Latn": "always toward better things",
        "fra_Latn": "toujours vers de meilleures choses",
        "spa_Latn": "siempre hacia cosas mejores",
        "deu_Latn": "immer zu besseren dingen",
        "ita_Latn": "sempre verso cose migliori",
        "por_Latn": "sempre para coisas melhores"
      }
    },
    {
      "latin": "semper eadem",
      "category": "phrase",
      "translations": {
        "eng_Latn": "always the same",
        "fra_Latn": "toujours la meme",
        "spa_Latn": "siempre la misma",
        "deu_Latn": "immer dieselbe",
        "ita_Latn": "sempre la stessa",
        "por_Latn": "sempre a mesma"
      }
    },
    {
      "latin": "semper excelsius",
      "category": "phrase",
      "translations": {
        "eng_Latn": "ever upward",
        "fra_Latn": "toujours plus haut",
        "spa_Latn": "siempre mas alto",
        "deu_Latn": "immer hoher",
        "ita_Latn": "sempre piu in alto",
        "por_Latn": "sempre mais alto"
      }
    },
    {
      "latin": "semper fidelis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "always faithful",
        "fra_Latn": "toujours fidele",
        "spa_Latn": "siempre fiel",
        "deu_Latn": "immer treu",
        "ita_Latn": "sempre fedele",
        "por_Latn": "sempre fiel"
      }
    },
    {
      "latin": "semper fortis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "always brave",
        "fra_Latn": "toujours courageux",
        "spa_Latn": "siempre valiente",
        "deu_Latn": "immer tapfer",
        "ita_Latn": "sempre coraggioso",
        "por_Latn": "sempre corajoso"
      }
    },
    {
      "latin": "semper paratus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "always ready",
        "fra_Latn": "toujours pret",
        "spa_Latn": "siempre listo",
        "deu_Latn": "immer bereit",
        "ita_Latn": "sempre pronto",
        "por_Latn": "sempre pronto"
      }
    },
    {
      "latin": "semper primus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "always first",
        "fra_Latn": "toujours premier",
        "spa_Latn": "siempre primero",
        "deu_Latn": "immer zuerst",
        "ita_Latn": "sempre primo",
        "por_Latn": "sempre primeiro"
      }
    },
    {
      "latin": "semper ubi sub ubi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "always where under where",
        "fra_Latn": "toujours ou sous ou",
        "spa_Latn": "siempre donde bajo donde",
        "deu_Latn": "immer wo unter wo",
        "ita_Latn": "sempre dove sotto dove",
        "por_Latn": "sempre onde sob onde"
      }
    },
    {
      "latin": "sermo animi imago est",
      "category": "phrase",
      "translations": {
        "eng_Latn": "speech is the image of the mind",
        "fra_Latn": "la parole est l image de l ame",
        "spa_Latn": "el habla es la imagen del alma",
        "deu_Latn": "rede ist das bild des geistes",
        "ita_Latn": "il discorso e immagine dell animo",
        "por_Latn": "a fala e a imagem da alma"
      }
    },
    {
      "latin": "serva me servabo te",
      "category": "phrase",
      "translations": {
        "eng_Latn": "save me and i will save you",
        "fra_Latn": "sauve moi et je te sauverai",
        "spa_Latn": "salvame y te salvare",
        "deu_Latn": "rette mich und ich werde dich retten",
        "ita_Latn": "salvami e ti salvero",
        "por_Latn": "salva me e eu te salvarei"
      }
    },
    {
      "latin": "si deus nobiscum quis contra nos",
      "category": "phrase",
      "translations": {
        "eng_Latn": "if god is with us who is against us",
        "fra_Latn": "si dieu est avec nous qui est contre nous",
        "spa_Latn": "si dios esta con nosotros quien contra nosotros",
        "deu_Latn": "wenn gott mit uns ist wer ist gegen uns",
        "ita_Latn": "se dio e con noi chi e contro di noi",
        "por_Latn": "se deus esta conosco quem sera contra nos"
      }
    },
    {
      "latin": "si monumentum requiris circumspice",
      "category": "phrase",
      "translations": {
        "eng_Latn": "if you seek his monument look around",
        "fra_Latn": "si tu cherches son monument regarde autour de toi",
        "spa_Latn": "si buscas su monumento mira a tu alrededor",
        "deu_Latn": "wenn du sein denkmal suchst schau um dich",
        "ita_Latn": "se cerchi il suo monumento guardati intorno",
        "por_Latn": "se procuras seu monumento olha ao redor"
      }
    },
    {
      "latin": "si quaeris peninsulam amoenam circumspice",
      "category": "phrase",
      "translations": {
        "eng_Latn": "if you seek a pleasant peninsula look about you",
        "fra_Latn": "si tu cherches une agreable peninsule regarde autour de toi",
        "spa_Latn": "si buscas una peninsula agradable mira a tu alrededor",
        "deu_Latn": "wenn du eine schone halbinsel suchst sieh dich um",
        "ita_Latn": "se cerchi una piacevole penisola guardati intorno",
        "por_Latn": "se procuras uma peninsula agradavel olha ao redor"
      }
    },
    {
      "latin": "sic itur ad astra",
      "category": "phrase",
      "translations": {
        "eng_Latn": "thus one goes to the stars",
        "fra_Latn": "ainsi l on va aux etoiles",
        "spa_Latn": "asi se va a las estrellas",
        "deu_Latn": "so geht man zu den sternen",
        "ita_Latn": "cosi si va alle stelle",
        "por_Latn": "assim se vai as estrelas"
      }
    },
    {
      "latin": "sic parvis magna",
      "category": "phrase",
      "translations": {
        "eng_Latn": "greatness from small beginnings",
        "fra_Latn": "grandeur a partir de petits commencements",
        "spa_Latn": "grandeza desde pequenos comienzos",
        "deu_Latn": "grosse aus kleinen anfangen",
        "ita_Latn": "grandezza da piccoli inizi",
        "por_Latn": "grandeza a partir de pequenos comecos"
      }
    },
    {
      "latin": "sic passim",
      "category": "phrase",
      "translations": {
        "eng_Latn": "thus everywhere",
        "fra_Latn": "ainsi partout",
        "spa_Latn": "asi por todas partes",
        "deu_Latn": "so uberall",
        "ita_Latn": "cosi dappertutto",
        "por_Latn": "assim por toda parte"
      }
    },
    {
      "latin": "silentium est aureum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "silence is golden",
        "fra_Latn": "le silence est d or",
        "spa_Latn": "el silencio es oro",
        "deu_Latn": "schweigen ist gold",
        "ita_Latn": "il silenzio e d oro",
        "por_Latn": "o silencio e ouro"
      }
    },
    {
      "latin": "similia similibus curentur",
      "category": "phrase",
      "translations": {
        "eng_Latn": "let like be cured by like",
        "fra_Latn": "que le semblable soit gueri par le semblable",
        "spa_Latn": "que lo semejante sea curado por lo semejante",
        "deu_Latn": "ahnliches moge durch ahnliches geheilt werden",
        "ita_Latn": "il simile sia curato dal simile",
        "por_Latn": "o semelhante seja curado pelo semelhante"
      }
    },
    {
      "latin": "sine ira et studio",
      "category": "phrase",
      "translations": {
        "eng_Latn": "without anger and partiality",
        "fra_Latn": "sans colere ni parti pris",
        "spa_Latn": "sin ira ni parcialidad",
        "deu_Latn": "ohne zorn und eifer",
        "ita_Latn": "senza ira e senza parzialita",
        "por_Latn": "sem ira e sem parcialidade"
      }
    },
    {
      "latin": "sine labore nihil",
      "category": "phrase",
      "translations": {
        "eng_Latn": "nothing without work",
        "fra_Latn": "rien sans travail",
        "spa_Latn": "nada sin trabajo",
        "deu_Latn": "nichts ohne arbeit",
        "ita_Latn": "nulla senza lavoro",
        "por_Latn": "nada sem trabalho"
      }
    },
    {
      "latin": "sine qua non",
      "category": "phrase",
      "translations": {
        "eng_Latn": "an indispensable condition",
        "fra_Latn": "condition indispensable",
        "spa_Latn": "condicion indispensable",
        "deu_Latn": "unerlassliche bedingung",
        "ita_Latn": "condizione indispensabile",
        "por_Latn": "condicao indispensavel"
      }
    },
    {
      "latin": "sol omnibus lucet",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the sun shines for everyone",
        "fra_Latn": "le soleil brille pour tous",
        "spa_Latn": "el sol brilla para todos",
        "deu_Latn": "die sonne scheint fur alle",
        "ita_Latn": "il sole splende per tutti",
        "por_Latn": "o sol brilha para todos"
      }
    },
    {
      "latin": "sola fide",
      "category": "phrase",
      "translations": {
        "eng_Latn": "by faith alone",
        "fra_Latn": "par la foi seule",
        "spa_Latn": "solo por la fe",
        "deu_Latn": "allein durch den glauben",
        "ita_Latn": "per sola fede",
        "por_Latn": "somente pela fe"
      }
    },
    {
      "latin": "sola dosis facit venenum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the dose alone makes the poison",
        "fra_Latn": "la dose seule fait le poison",
        "spa_Latn": "solo la dosis hace el veneno",
        "deu_Latn": "allein die dosis macht das gift",
        "ita_Latn": "solo la dose fa il veleno",
        "por_Latn": "só a dose faz o veneno"
      }
    },
    {
      "latin": "soli deo gloria",
      "category": "phrase",
      "translations": {
        "eng_Latn": "glory to god alone",
        "fra_Latn": "gloire a dieu seul",
        "spa_Latn": "gloria solo a dios",
        "deu_Latn": "allein gott die ehre",
        "ita_Latn": "gloria a dio solo",
        "por_Latn": "gloria somente a deus"
      }
    },
    {
      "latin": "solvitur ambulando",
      "category": "phrase",
      "translations": {
        "eng_Latn": "it is solved by walking",
        "fra_Latn": "cela se resout en marchant",
        "spa_Latn": "se resuelve caminando",
        "deu_Latn": "es lost sich im gehen",
        "ita_Latn": "si risolve camminando",
        "por_Latn": "resolve se caminhando"
      }
    },
    {
      "latin": "spes bona",
      "category": "phrase",
      "translations": {
        "eng_Latn": "good hope",
        "fra_Latn": "bonne esperance",
        "spa_Latn": "buena esperanza",
        "deu_Latn": "gute hoffnung",
        "ita_Latn": "buona speranza",
        "por_Latn": "boa esperanca"
      }
    },
    {
      "latin": "spes mea in deo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "my hope is in god",
        "fra_Latn": "mon espoir est en dieu",
        "spa_Latn": "mi esperanza esta en dios",
        "deu_Latn": "meine hoffnung ist in gott",
        "ita_Latn": "la mia speranza e in dio",
        "por_Latn": "minha esperanca esta em deus"
      }
    },
    {
      "latin": "spiritus mundi",
      "category": "phrase",
      "translations": {
        "eng_Latn": "spirit of the world",
        "fra_Latn": "esprit du monde",
        "spa_Latn": "espiritu del mundo",
        "deu_Latn": "geist der welt",
        "ita_Latn": "spirito del mondo",
        "por_Latn": "espirito do mundo"
      }
    },
    {
      "latin": "status quo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "the existing state of affairs",
        "fra_Latn": "l etat actuel des choses",
        "spa_Latn": "el estado actual de las cosas",
        "deu_Latn": "der bestehende zustand",
        "ita_Latn": "lo stato delle cose",
        "por_Latn": "o estado atual das coisas"
      }
    },
    {
      "latin": "sub rosa",
      "category": "phrase",
      "translations": {
        "eng_Latn": "in secret",
        "fra_Latn": "en secret",
        "spa_Latn": "en secreto",
        "deu_Latn": "unter der rose",
        "ita_Latn": "in segreto",
        "por_Latn": "em segredo"
      }
    },
    {
      "latin": "sub specie aeternitatis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "under the aspect of eternity",
        "fra_Latn": "sous l aspect de l eternite",
        "spa_Latn": "bajo el aspecto de la eternidad",
        "deu_Latn": "unter dem gesichtspunkt der ewigkeit",
        "ita_Latn": "sotto l aspetto dell eternita",
        "por_Latn": "sob o aspecto da eternidade"
      }
    },
    {
      "latin": "sub verbo",
      "category": "phrase",
      "translations": {
        "eng_Latn": "under the word",
        "fra_Latn": "sous la parole",
        "spa_Latn": "bajo la palabra",
        "deu_Latn": "unter dem wort",
        "ita_Latn": "sotto la parola",
        "por_Latn": "sob a palavra"
      }
    },
    {
      "latin": "summa cum laude",
      "category": "phrase",
      "translations": {
        "eng_Latn": "with highest praise",
        "fra_Latn": "avec la plus haute louange",
        "spa_Latn": "con el mayor elogio",
        "deu_Latn": "mit hochstem lob",
        "ita_Latn": "con somma lode",
        "por_Latn": "com a mais alta honra"
      }
    },
    {
      "latin": "summum bonum",
      "category": "phrase",
      "translations": {
        "eng_Latn": "highest good",
        "fra_Latn": "souverain bien",
        "spa_Latn": "sumo bien",
        "deu_Latn": "hochstes gut",
        "ita_Latn": "sommo bene",
        "por_Latn": "sumo bem"
      }
    },
    {
      "latin": "sursum corda",
      "category": "phrase",
      "translations": {
        "eng_Latn": "lift up your hearts",
        "fra_Latn": "elevons nos coeurs",
        "spa_Latn": "levantemos el corazon",
        "deu_Latn": "erhebet die herzen",
        "ita_Latn": "in alto i cuori",
        "por_Latn": "ao alto os coracoes"
      }
    },
    {
      "latin": "tabula rasa",
      "category": "phrase",
      "translations": {
        "eng_Latn": "blank slate",
        "fra_Latn": "table rase",
        "spa_Latn": "tabla rasa",
        "deu_Latn": "unbeschriebenes blatt",
        "ita_Latn": "tabula rasa",
        "por_Latn": "tabula rasa"
      }
    },
    {
      "latin": "tempora mutantur et nos mutamur in illis",
      "category": "phrase",
      "translations": {
        "eng_Latn": "times change and we change with them",
        "fra_Latn": "les temps changent et nous changeons avec eux",
        "spa_Latn": "los tiempos cambian y nosotros cambiamos con ellos",
        "deu_Latn": "die zeiten andern sich und wir uns in ihnen",
        "ita_Latn": "i tempi cambiano e noi cambiamo con essi",
        "por_Latn": "os tempos mudam e nos mudamos com eles"
      }
    },
    {
      "latin": "tempus fugit",
      "category": "phrase",
      "translations": {
        "eng_Latn": "time flies",
        "fra_Latn": "le temps fuit",
        "spa_Latn": "el tiempo vuela",
        "deu_Latn": "die zeit fliegt",
        "ita_Latn": "il tempo fugge",
        "por_Latn": "o tempo voa"
      }
    },
    {
      "latin": "terra incognita",
      "category": "phrase",
      "translations": {
        "eng_Latn": "unknown land",
        "fra_Latn": "terre inconnue",
        "spa_Latn": "tierra desconocida",
        "deu_Latn": "unbekanntes land",
        "ita_Latn": "terra incognita",
        "por_Latn": "terra desconhecida"
      }
    },
    {
      "latin": "timeo danaos et dona ferentes",
      "category": "phrase",
      "translations": {
        "eng_Latn": "i fear the greeks even when bearing gifts",
        "fra_Latn": "je crains les grecs meme quand ils portent des cadeaux",
        "spa_Latn": "temo a los griegos incluso cuando traen regalos",
        "deu_Latn": "ich furchte die griechen auch wenn sie geschenke bringen",
        "ita_Latn": "temo i greci anche quando portano doni",
        "por_Latn": "temo os gregos ate quando trazem presentes"
      }
    },
    {
      "latin": "totus tuus",
      "category": "phrase",
      "translations": {
        "eng_Latn": "totally yours",
        "fra_Latn": "tout a toi",
        "spa_Latn": "totalmente tuyo",
        "deu_Latn": "ganz dein",
        "ita_Latn": "tutto tuo",
        "por_Latn": "todo teu"
      }
    }
  ]
}
"""


def normalize_phrase_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def normalize_language_key(text: str) -> str:
    return normalize_phrase_key(text).replace("-", "_").replace(" ", "_")


SCRIPT_NAMES: Dict[str, str] = {
    "Arab": "Arabic script",
    "Armn": "Armenian script",
    "Beng": "Bengali script",
    "Cyrl": "Cyrillic script",
    "Deva": "Devanagari script",
    "Ethi": "Ethiopic script",
    "Geor": "Georgian script",
    "Grek": "Greek script",
    "Gujr": "Gujarati script",
    "Guru": "Gurmukhi script",
    "Hang": "Hangul script",
    "Hans": "Simplified script",
    "Hant": "Traditional script",
    "Hebr": "Hebrew script",
    "Jpan": "Japanese script",
    "Khmr": "Khmer script",
    "Knda": "Kannada script",
    "Laoo": "Lao script",
    "Latn": "Latin script",
    "Mlym": "Malayalam script",
    "Mymr": "Myanmar script",
    "Orya": "Odia script",
    "Sinh": "Sinhala script",
    "Taml": "Tamil script",
    "Telu": "Telugu script",
    "Tfng": "Tifinagh script",
    "Thai": "Thai script",
    "Tibt": "Tibetan script",
}

LANGUAGE_NAME_OVERRIDES: Dict[str, str] = {
    "ace": "Acehnese",
    "ajp": "South Levantine Arabic",
}

LANGUAGE_ALIAS_OVERRIDES: Dict[str, str] = {
    "arabic": "arb_Arab",
    "chinese": "zho_Hans",
    "chinese_simplified": "zho_Hans",
    "chinese_traditional": "zho_Hant",
    "english": "eng_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "greek": "ell_Grek",
    "italian": "ita_Latn",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "latin": "lat_Latn",
    "norwegian": "nob_Latn",
    "persian": "pes_Arab",
    "portuguese": "por_Latn",
    "spanish": "spa_Latn",
}

MANUAL_LANGUAGE_ENTRIES: List[Dict[str, str]] = [
    {
        "base_code": "lat",
        "code": "lat_Latn",
        "name": "Latin",
        "script": "Latin script",
    }
]


def _resolve_language_name(base_code: str) -> str:
    if base_code in LANGUAGE_NAME_OVERRIDES:
        return LANGUAGE_NAME_OVERRIDES[base_code]

    language = pycountry.languages.get(alpha_3=base_code)
    if language is not None:
        return getattr(language, "name", base_code)

    return base_code


@lru_cache(maxsize=1)
def list_language_entries() -> List[Dict[str, str]]:
    base_code_counts = Counter(code.split("_")[0] for code in FAIRSEQ_LANGUAGE_CODES)
    entries: List[Dict[str, str]] = []

    for code in FAIRSEQ_LANGUAGE_CODES:
        base_code, script_code = code.split("_", 1)
        language_name = _resolve_language_name(base_code)
        script_name = SCRIPT_NAMES.get(script_code, script_code)

        if base_code_counts[base_code] > 1:
            display_name = f"{language_name} ({script_name})"
        else:
            display_name = language_name

        entries.append(
            {
                "base_code": base_code,
                "code": code,
                "name": display_name,
                "script": script_name,
            }
        )

    entries.extend(MANUAL_LANGUAGE_ENTRIES)
    return entries


@lru_cache(maxsize=1)
def get_language_entry_map() -> Dict[str, Dict[str, str]]:
    return {entry["code"]: entry for entry in list_language_entries()}


@lru_cache(maxsize=1)
def build_language_alias_map() -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    entries = list_language_entries()
    base_code_counts = Counter(entry["base_code"] for entry in entries)

    for entry in entries:
        code = entry["code"]
        base_name = _resolve_language_name(entry["base_code"])
        alias_map[normalize_language_key(code)] = code
        alias_map[normalize_language_key(entry["name"])] = code

        if base_code_counts[entry["base_code"]] == 1:
            alias_map[normalize_language_key(base_name)] = code

    alias_map.update(LANGUAGE_ALIAS_OVERRIDES)
    return alias_map


@lru_cache(maxsize=1)
def load_phrase_database() -> Dict[str, Any]:
    payload = json.loads(PHRASE_DATABASE_JSON)
    payload["languages"] = list_language_entries()
    payload["metadata"] = {
        "base_language_count": len({entry["base_code"] for entry in payload["languages"]}),
        "language_entry_count": len(payload["languages"]),
        "curated_phrase_translation_count": len(payload["phrases"]),
        "notes": "Language entries are sourced from the official NLLB tokenizer list bundled with transformers, plus a manual Latin entry for the package default source flow.",
    }
    return payload


@lru_cache(maxsize=1)
def build_phrase_index() -> Dict[str, Dict[str, Any]]:
    phrases = load_phrase_database()["phrases"]
    return {normalize_phrase_key(entry["latin"]): entry for entry in phrases}


def lookup_phrase(text: str, target_language: str) -> Optional[str]:
    entry = build_phrase_index().get(normalize_phrase_key(text))
    if entry is None:
        return None
    return entry["translations"].get(target_language)


def get_phrase_entry(text: str) -> Optional[Dict[str, Any]]:
    entry = build_phrase_index().get(normalize_phrase_key(text))
    if entry is None:
        return None
    return entry.copy()


def list_phrase_entries() -> List[Dict[str, Any]]:
    return list(load_phrase_database()["phrases"])


def list_latin_phrases() -> List[str]:
    return [entry["latin"] for entry in load_phrase_database()["phrases"]]


def list_phrase_languages() -> List[Dict[str, str]]:
    return list(load_phrase_database()["languages"])
