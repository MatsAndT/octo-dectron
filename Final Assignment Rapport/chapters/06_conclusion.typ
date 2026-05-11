= Konklusjon <Konklusjon>

== Konklusjon

Prosjektet undersøkte om maskinlæringsmodeller kan klassifisere dronemodus basert på RF-signaler fra DroneRF-datasettet. Resultatet er at begge modellene klarer oppgaven godt, men med ulike styrker og svakheter.

MLP-modellen oppnådde en testnøyaktighet på 82,61 % og en macro-F1 på 0,82. Den avgjørende forbedringen kom ikke fra arkitekturen, men fra en bedre signalrepresentasjon: ved å bytte fra én enkelt FFT per opptak til 192 statistisk sammensatte frekvensbånds-statistikker fikk modellen langt mer informasjon å arbeide med. Den tyngste regulariseringen (α = 1,0) og den enkleste arkitekturen (ett lag med 64 nevroner) vant konsekvent i hyperparameterletingen.

CNN-modellen oppnådde 71,7 % testnøyaktighet og macro-F1 på 0,66. Modellen klassifiserte tre av fem klasser godt, men slet med modus 2 og 3. BUI-koden slår sammen opptak fra ulike droneprodusenter, noe som øker variasjonen innad i klassen og gjør dem vanskeligere å lære.

Prosjektet bekrefter at RF-signaler inneholder tilstrekkelig informasjon til å klassifisere dronemodus med god treffsikkerhet, og at valg av signalrepresentasjon er viktigere enn valg av modelltype for dette datasettet.

== Veien videre

Klassene med lavest ytelse — modus 2 og 3 for CNN — indikerer at det å klassifisere modus på tvers av ulike produsenter er problematisk. En naturlig forbedring ville være å dele klassene per dronefamilie, slik at modellen ikke forsøker å lære én felles signatur for signaler fra ulike produsenter. Dette ville kreve en annen klasseinndeling i BUI-koden.

En annen forbedring er å øke datasettets størrelse. Med 227 opptak og fem klasser er treningssettet knapt, og modellene er sterkt begrenset av dette. Flere opptak — særlig av modus 2 og 3 — ville trolig gi bedre og mer stabile resultater, og gjøre det mulig å teste om en større CNN gir bedre ytelse enn 11 000 parametere.

Et aspekt vi ikke har undersøkt er vindus-sensitivitet: 64 000 sampler per vindu er ett bestemt valg, og det er uklart om kortere eller lengre vinduer ville gitt bedre klassifisering. Kortere vinduer gir finere tidsoppløsning men svakere frekvensoppløsning, mens lengre vinduer gir det motsatte. En systematisk gjennomkjøring av ulike vindus-størrelser kunne avklart dette.

For praktisk bruk er det også relevant å vurdere sanntidsbruk og robusthet. Med 64 000 sampler per vindu og 50 % overlapping tar det et par sekunder å fylle ett vindu ved typiske samplerater for DroneRF-datasettet. I en reell detektor er latens viktig, og en analyse av minste vindus-størrelse som gir tilfredsstillende ytelse ville vært nyttig. I tillegg er datasettet samlet inn i kontrollerte omgivelser, og det er uvisst hvordan modellene ville prestert i et reelt RF-miljø med WiFi-interferens og andre forstyrrelser i 2,4 GHz-båndet.


