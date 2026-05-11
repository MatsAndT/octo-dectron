#set text(lang: "NB")
#import "conf.typ": conf
#set par(justify: true)
#import "@preview/wordometer:0.1.5": word-count, total-words

// Global state-variabel for ordtelling kap 1-6
#let chapter-words = state("chapter-words", 0)

#show: doc => conf(
  doc-total-words: context chapter-words.final(),
  title: [
    Avsluttende oppgave
  ],
  className: [
    ING3513 Introduksjon til kunstig intelligens og maskinlæring
  ],
  abstract: "Dette prosjektet undersøker om maskinlæringsmodeller kan gjenkjenne dronemodus ut fra RF-signaler i DroneRF-datasettet. Datasettet har 227 opptak i fem klasser fra tre dronetyper. Til dette benyttes det to maskinlæringsmodeller: MLP og CNN. Begge modellene deler hvert rå-signal opp i korte, overlappende vinduer og beregner frekvensenergi (FFT) for hvert vindu, slik at man får en tidsserie av frekvensspekteret. MLP-en summerer hver serie til 192 målbare trekk (gjennomsnitt, standardavvik og maksimum per bånd), mens CNN-en bruker hele vindussekvensen direkte. MLP-en nådde 82,6 % testnøyaktighet og macro-F1 0,82; CNN-en 71,7 % og macro-F1 0,66. Resultatene viser at RF-signaler inneholder nok informasjon til å klassifisere dronemodus, og at hvordan signalet representeres er viktigere enn hvilken modelltype som brukes for dette datasettet.",
  doc,
)

/*
Sammendrag
*/
#show: word-count.with(exclude: (footnote, raw, figure))

// Tel kap 1-6, oppdater state INNE i blokken
#word-count(total => [
  #chapter-words.update(total.words)

  #include "chapters/01_introduction.typ"
  #pagebreak()
  #include "chapters/02_dataset.typ"
  #pagebreak()
  #include "chapters/03_method.typ"
  #pagebreak()
  #include "chapters/04_results.typ"
  #pagebreak()
  #include "chapters/05_discussion.typ"
  #pagebreak()
  #include "chapters/06_conclusion.typ"
  #pagebreak()
], exclude: (footnote, raw, figure))

#word-count(total => [
  = Litteraturreferanser
  #bibliography(
    "referanser.yml",
    title: none,
    style: "institute-of-electrical-and-electronics-engineers"
  )
  #pagebreak()
  #include "chapters/appendix.typ"
])