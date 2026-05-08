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
  abstract: "lorem ipsum",
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

