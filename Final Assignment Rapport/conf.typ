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
  header: context align(right, [#image("./UGRADERT.png", width: 20%)])
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
  align(center)[#h(1fr) Adrianne Bendiksen#h(1fr) Stian Sivertsen Loddengaard#h(1fr) Robert Brenner Marthins#h(1fr)]
  v(15pt)
  align(center)[#h(1fr)  Mads Trøen#h(1fr) Mats Andreas Tønnesland#h(1fr) Nils Christian Halvorsen Wikstrøm#h(1fr)]
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
    Antall ord: #doc-total-words\
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

    
    *Sammendrag* \
      #abstract \
    ]

    set page(
    paper: "a4",
    numbering: "i"
    )
    counter(page).update(2)

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
