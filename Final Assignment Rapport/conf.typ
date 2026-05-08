#let today = datetime.today()

#let conf(
  title: none,
  className: none,
  labNr: none,
  authors: (),
  writerNr: none,
  abstract: [],
  doc,
  references: [],
  doc-total-words: none,
) = {

   set page(
  header: [
    #text(fill: rgb("00AA00"))[
      #h(1fr)
      /**UGRADERT**/
    ]
  ]
)
  
  show figure.where(
    kind: table
  ): set figure.caption(position: top)

  align(center, figure(image("/img/CISK_logo.png", width: 17%)))
  par(justify: false)[
    #h(1fr)
    #today.display("[day].[month].[year]")
  ]

  

    
  line(length: 100%)

  set align(center)
  par(justify: true)[
    \
    #text(20pt, title) \
    \
    Drone klassifisering fra RF-signaler
    \
    \
  ]
  line(length: 100%)
  par(justify: true)[
    \ \
    Av\
    \
  ]

  v(15pt)
  align(center)[#h(1fr) Stian Sivertsen Loddengaard#h(1fr) Robert Brenner Marthins#h(1fr) Adrianne Bendiksen#h(1fr)]
  v(15pt)
  align(center)[#h(1fr)  Mats Andreas Tønnesland#h(1fr) Mads Trøen#h(1fr) Nils Christian Halvorsen Wikstrøm#h(1fr)]
  v(15pt)


  par(justify: true)[
    \
    \
  ]
  
  line(length: 100%)
  par(justify: true)[
    \
    \
    #set align(left)
    Ving 77\
    \
    \
    \
    \
    Words: 1\
    \
    \
  ]
  
  
  
  line(length: 100%)
  par(justify: true)[
    \ 
    \
    \
  ]

  set page(
    paper: "a4",
    numbering: none
    )
    counter(page).update(0)



  if abstract != [] {
    par(justify: false)[
      \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ [Blank med hensikt] \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
      ]
  }

  set page(
    paper: "a4",
    numbering: "i"
    )
    counter(page).update(1)

  align(left)[
    = Midlertidig notater for skriving:
    - Plan for prosjektet som sendt til Øystein:
 
Vi har delt inn i to grupper. En gruppe har hovedfokus på skriving. Den andre gruppen har hovedfokus på programmering. Disse gruppene overlapper litt med tanke på at skrivegruppen må ha forståelse for hva som foregår på kodesiden og motsatt. Vi deler opp på denne måten for å unngå for mange conflicts på GitHub, og for at alle skal få noe å gjøre.

Vi planlegger å bruke og sammenlignge klassifisering med MLP og CNN. Vi kommer til å starte med å lese igjennom det vi har fått av ressurser slik at alle har en grunnleggende forståelse av hva vi skal jobbe med. Deretter skal vi utvide denne forståelsen ved å visualisere og forstå datasettet vi har fått. Vi bruker felles dokument for skriving og felles GitHub-res slik at alle kan holde seg oppdatert og forstå hvor vi er i løypa
    - Vi har bestemt oss for å forsøke å klassifisere basert dronen sin modus, ikke dronetype.
      - Her har vi sikkert en del å yappe om på diskusjon. Hvorfor er det lurt? Hva kan være bakdelen? Er det mulighet for fremtidig arbeid som kan finne ut av type OG modus
    #pagebreak()
    
    *Sammendrag* \
      #abstract \
    ]

    set page(
    paper: "a4",
    numbering: "i"
    )
    counter(page).update(2)

  pagebreak()

  include "chapters/07_prologue.typ"

  pagebreak()
  

  set heading(outlined: false)
  include "chapters/00_abbreviations.typ"
  set heading(outlined: true)


  
  pagebreak()


  
  outline(
      title: "Innholdsfortegnelse",
      indent: auto,
      depth: 3,
    )
  set heading(numbering: "1.")
  set page(numbering: "1 / 1")
  counter(page).update(1)
  set align(left)
  
  doc   
}
