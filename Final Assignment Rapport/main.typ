#set text(lang: "NB")
#import "conf.typ": conf
#set par(justify: true)

#import "@preview/wordometer:0.1.5": word-count, total-words

#show: doc => conf(
  doc-total-words: [1], /* Anntal ord fra innledning start, se show: word-count sin plasering */
  title: [
    Final assignment
  ],
  className: [
    ING3513 Introduksjon til kunstig intelligens og maskinlæring
  ],
  abstract: "Dette prosjektet undersøker om maskinlæringsmodeller kan klassifisere dronemodus basert på RF-signaler fra DroneRF-datasettet. Datasettet inneholder 227 opptak fordelt over fem klasser fra tre dronemodeller. Begge modellene bruker glidende vinduisering av råsignalene: for hvert vindu beregnes frekvensbånds-energier via FFT, noe som gir en tidsserie av spektrale profiler. MLP komprimerer disse til 192 aggregerte egenskaper (gjennomsnitt, standardavvik og maksimum per bånd), mens CNN behandler vinduesekvensen direkte. MLP oppnådde 82,6 % testnøyaktighet og macro-F1 på 0,82; CNN oppnådde 71,7 % og macro-F1 på 0,66. Begge modellene feiler på de samme to klassene, der BUI-koden slår sammen opptak fra ulike dronefamilier. Resultatene viser at RF-signaler inneholder tilstrekkelig informasjon til å klassifisere dronemodus, og at valg av signalrepresentasjon er viktigere enn valg av modelltype for dette datasettet.",
  doc,
)

/*
Sammendrag
*/

#show: word-count.with(exclude: (<ruter-konfigurasjon>, raw, figure))

#word-count(total => [
  #word-count(total => [
    #include "chapters/01_introduction.typ"
  ], exclude: (footnote, raw, figure))
  
  #pagebreak()

  #word-count(total => [
    #include "chapters/02_dataset.typ"
  ], exclude: (footnote, raw, figure))
  
  #pagebreak()
  
  #word-count(total => [
    #include "chapters/03_method.typ" 
  ], exclude: (footnote, raw, figure))
  
  #pagebreak()
  
  #word-count(total => [
    #include "chapters/04_results.typ"
  ], exclude: (footnote, raw, figure))
  
  #pagebreak()
  
  #word-count(total => [
    #include "chapters/05_discussion.typ"
  ], exclude: (footnote, raw, figure))
  
  #pagebreak()
  
  #word-count(total => [  
    #include "chapters/06_conclusion.typ"
  ], exclude: (footnote, raw, figure))
  
  #pagebreak()
], exclude: (footnote, raw, figure))



/*
*Her er eksempler:*
\
Hei
- det var
#figure(
  table(
  columns: (auto, auto, auto, auto),
  align: center,
  table.header([*Komponenttype*],[*Modell*],[*Serienummer*],[*Antall*]),

  [Spenningskilde],[Gwinstek GPS-3030],[834966],[1],
  [],[],[],[],)
, caption: [
  Instrumentliste
],
supplement: [Tabell]
)

@Leksjoner_kretsteknikk // Setter inn litteraturreferanse

#figure(
  image("img/placholder.png", width: 90%),
  caption: [
    Krets 1
  ],
  supplement: [Figur]
) <krets1> // Gir en referanse til figuren bruk: @krets1 for å få ut figurnummeret, eks. Figur 1

*/


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

