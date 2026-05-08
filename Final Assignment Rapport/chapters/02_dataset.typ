= Dataset <Teori>
I dette prosjektet bruker vi DroneRF-datasettet, som består av radiofrekvensopptak fra kommersielle droner under kontrollerte forhold. Datasettet ble publisert sammen med artikkelen til Al-Sa'd et al., der formålet var å bygge en åpen database for RF-basert dronedeteksjon og droneidentifikasjon. Datasettet inneholder opptak fra tre dronetyper: Parrot Bebop, Parrot AR Drone og DJI Phantom. I tillegg inneholder datasettet bakgrunnsopptak uten aktiv drone.

Hvert opptak er merket med en BUI-kode, som er en bitstreng som beskriver både dronetypen og tilstanden til dronen. De første bitene identifiserer dronen, mens de siste bitene beskriver dronens modus. I vårt prosjekt brukes disse kodene til å lage tre mulige klassifikasjonsmål: `target_binary`, `target_family` og `target_mode`. `target_binary` skiller mellom bakgrunn og drone, `target_family` skiller mellom dronefamilier, og `target_mode` beskriver en mer detaljert 10-klasse klassifisering av dronemodus. Hovedfokuset i prosjektet er `target_mode`.

Dronene i datasettet kommuniserer i 2.4 GHz-båndet. Siden én mottaker ikke kunne dekke hele signalbåndet samtidig, ble opptakene gjort med to RF-mottakere. Den ene mottakeren registrerte den lave delen av frekvensbåndet, merket `L`, mens den andre registrerte den høye delen, merket `H`. Datasettene inneholder derfor ofte to tilhørende filer for samme opptak: én for lavt bånd og én for høyt bånd.

Hver fil inneholder rå tidsdomenedata lagret som CSV-verdier. En typisk fil kan for eksempel hete `10100L_3.csv`. Denne filen kan tolkes som et opptak fra dronen identifisert av `101`, i modus `00`, fra lavfrekvensbåndet `L`, og som segment nummer 3 i opptakssekvensen. På denne måten inneholder filnavnet både label-informasjon og informasjon om hvilken del av RF-opptaket filen tilhører.

== Datakvalitet og utfordringer

DroneRF-datasettet har flere egenskaper som gjør det utfordrende å bruke direkte i maskinlæring. For det første består opptakene av svært lange tidsserier. En enkelt CSV-fil inneholder mange millioner målepunkter, og det er derfor ikke praktisk å bruke hele filer direkte som individuelle treningspunkter. Signalene må deles opp i mindre vinduer før de kan mates inn i modellene.

For det andre er datasettet høy-dimensjonalt. Selv korte vinduer av rå RF-data kan inneholde tusenvis av verdier. Dette gir mye informasjon, men øker også risikoen for overtilpasning og gjør modelltreningen mer krevende. Derfor må vi velge en representasjon som balanserer informasjonsinnhold og beregningskostnad.

En tredje utfordring er klasseubalanse. Noen droner og moduser forekommer flere ganger enn andre. Dette kan gjøre at en modell lærer å prioritere majoritetsklassene, samtidig som den gjør det dårligere på klasser med færre eksempler. Av den grunn evaluerer vi ikke modellene kun med accuracy, men også med macro-F1 og confusion matrix.

En fjerde utfordring er at enkelte klasser kan være overlappende i RF-mønster. To ulike moduser kan gi ganske like radiosignaturer, særlig dersom de tilhører samme dronefamilie eller bruker samme kommunikasjonsprotokoll. Dette gjør modusklassifisering vanskeligere enn ren drone/ikke-drone-deteksjon. Den opprinnelige DroneRF-studien viser også at klassifikasjonsytelsen synker når antall klasser øker, noe som tyder på at de mer detaljerte klassene er vanskeligere å skille.



== Utvelging av features fra datasettet
  Hvert opptak er knyttet til en BUI-kode der de tre første bitene identifiserer
  maskinvaren og de to siste beskriver dronens modus:

  - Modus 1 (00): Påslått og tilkoblet kontroller.
  - Modus 2 (01): Automatisk sveveflyging.
  - Modus 3 (10): Flyging uten videoopptak.
  - Modus 4 (11): Flyging med videoopptak.

#image("../images/bui.png")
  
Dronene kommuniserer over WiFi#sym.trademark på 2,4 GHz, med en total båndbredde på opptil 80 MHz ifølge @DroneRF_dataset. Siden én mottaker kun hadde 40 MHz øyeblikkelig båndbredde, ble to mottakere benyttet parallelt — én for lavt bånd (L) og én for høyt bånd (H). Med 40 MHz båndbredde registrerer mottakeren 40 000 000 målinger per sekund. Siden hver fil inneholder 10 000 000 verdier, representerer én fil 0,25 sekunder med opptak.

I @time_11000L_0 og @time_11000H_0 vises råsignalet i tidsdomenet for BUI 11000, for henholdsvis lavt og høyt bånd. Signalene inneholder kun informasjonssignalet rundt bærefrekvensen, selve bærefrekvensen er ikke synlig i tidsdomenet. Vi ser også at det lave båndet har høyere amplitude i dette opptaket.

#figure(
  image("../images/11000L_0.png"),
  caption: [Plott av 11000L_0.csv i tidsdomenet]
)<time_11000L_0>
#figure(
image("../images/11000H_0.png"),
caption: [Plott av 11000H_0.csv i tidsdomenet]
)<time_11000H_0>

Det er ikke nødvendig å rekonstruere ett komplett RF-signal fra de to båndene for maskinlæringsformål. Modellen kan fint finne struktur i de separate signalene, så lenge alle opptak er gjort med identisk utstyr og format, noe som er tilfellet i dette datasettet.

En mer robust tilnærming enn tidsdomenet er å se på frekvenssignaturen til hver dronetype og modus. Tidsdomene-signalet forventes å variere betydelig fra opptak til opptak ettersom informasjonen sendes til ulike tidspunkt, mens frekvensinnholdet er mer stabilt siden dronene kommuniserer på produsentspesifiserte frekvenser. Dronekommunikasjon er en form for frekvensmodulasjon, noe som betyr at informasjonen er kodet i frekvensene.

For å hente ut frekvensinnholdet bruker vi Rask Fourier-transformasjon (FFT). Det er imidlertid ikke mulig å rekonstruere det faktiske RF-frekvensspekteret fra disse opptakene. CSV-filene inneholder kun den reelle delen av signalet, ikke den imaginære. Uten en komplett In-fase/Kvadratur (IQ)-representasjon kan vi ikke skille mellom positive og negative frekvenser, de negative frekvensene brettes derfor over til den positive siden i FFT-resultatet. Det vi får er et basebånd-frekvensspekter fra 0 til 20 MHz, der vi ikke kan vite hvilke frekvenser som er brettet og hvilke ikke. Sammenligningen mellom tidsdomenet og frekvensdomenet for filen `11000L_0.csv` er vist i @freq_1000L_0.

#figure(
  image("../img/11000L_0_freq.png", width: 90%),
  caption: [Plott av tids- og frekvensdomene for 11000L_0]
)<freq_1000L_0>

Dette betyr at vi ikke kan lese av et fysisk meningsfylt RF-spekter, men all frekvensinformasjon er likevel bevart i en konsistent form. For modellen er dette tilstrekkelig: den trenger ikke forstå den fysiske betydningen av frekvensene, kun at alle segmenter presenteres på nøyaktig samme måte under trening og testing — noe som er garantert siden alle opptak bruker identisk utstyr og format.

For å utnytte informasjonen i frekvensspekteret best mulig, behandles inndataene ulikt avhengig av hvilken arkitektur som skal utføre klassifiseringen. For CNN-modellen velger vi å konkatenerere frekvensspekteret fra både det lave og høye båndet til én sammenhengende vektor. Logikken bak dette er at en CNN er spesialisert på å identifisere komplekse mønstre og hierarkiske sammenhenger i dataene. Ved å presentere hele det ubehandlede spekteret samlet, får modellen mulighet til å selv lære seg hvilke unike kombinasjoner av frekvenser og støysignaturer på tvers av de to båndene som er mest karakteristiske for hver enkelt dronemodus.

For MLP-modellen er strategien derimot å forenkle inndataene gjennom målrettet feature-ekstraksjon. Siden en MLP ikke har den samme innebygde evnen som en CNN til å filtrere ut struktur fra store, støyfulle datasett, velger vi å trekke ut de fem mest dominerende spektrale toppene med tilhørende frekvensindekser fra hvert bånd. Disse toppene fungerer som en konsis signatur av dronens kommunikasjon, da de ofte korresponderer med de faktiske bærefrekvensene som brukes til kontroll og videooverføring. Ved i tillegg å inkludere statistiske mål som minimum, maksimum, gjennomsnitt og standardavvik, får modellen viktig kontekst om signalets generelle energinivå og varians. Dette virker som et hensiktsmessig, ikke for komplekst feature-sett for MLP på såpass kompleks og sammensatt data som RF-signalet er.

#figure(
  caption: [Oversikt over de 28 originale trekkene (features) utvalgt for MLP-modellen],
  table(
    columns: (1fr, 1fr, 2fr),
    inset: 7pt,
    align: (left, left, left),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(40%) },
    
    [*Kategori*], [*Feature-navn*], [*Beskrivelse*],
    
    // Lavt bånd (L) - Topper
    table.cell(rowspan: 2, align: horizon)[*Spektrale topper (L)*], 
    [fL_peak1 -- fL_peak5], [Magnituden til de 5 største frekvenstoppene i lavt bånd.],
    [fL_freq1 -- fL_freq5], [Frekvensindeksene (0--2047) til de 5 største toppene i lavt bånd.],
    
    // Høyt bånd (H) - Topper
    table.cell(rowspan: 2, align: horizon)[*Spektrale topper (H)*], 
    [fH_peak1 -- fH_peak5], [Magnituden til de 5 største frekvenstoppene i høyt bånd.],
    [fH_freq1 -- fH_freq5], [Frekvensindeksene (0--2047) til de 5 største toppene i høyt bånd.],
    
    // Statistikk L
    table.cell(rowspan: 4, align: horizon)[*Statistikk (L)*],
    [fL_min], [Minimumsverdi i lavt bånd (støygulv).],
    [fL_max], [Maksimumsverdi i lavt bånd],
    [fL_mean], [Gjennomsnittlig energi i lavt bånd.],
    [fL_std], [Standardavvik (spredning) i lavt bånd.],
    
    // Statistikk H
    table.cell(rowspan: 4, align: horizon)[*Statistikk (H)*],
    [fH_min], [Minimumsverdi i høyt bånd (støygulv).],
    [fH_max], [Maksimumsverdi i høyt bånd],
    [fH_mean], [Gjennomsnittlig energi i høyt bånd.],
    [fH_std], [Standardavvik (spredning) i høyt bånd.],
  )
) <original_features_table>


== Eksplorativ Dataanalyse (EDA)
#figure(
  caption: [Statistisk oversikt over utvalgte MLP-features (N=227). Tabellen viser de to mest dominerende frekvenstoppene, samt aggregert statistikk for båndene.],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    inset: 5pt,
    align: (left, right, right, right, right, right, right, right),
    stroke: none,
    fill: (x, y) => if y == 0 { gray.lighten(50%) } else if calc.even(y) { gray.lighten(90%) },
    
    // Header
    [*Feature*], [*Mean*], [*Std*], [*Min*], [*25%*], [*50%*], [*75%*], [*Max*],
    
    // Frekvensindekser (Kun 1 og 2)
    table.cell(colspan: 8, [*Frekvensindekser (Indeks 0-2048)*]),
    [fL_freq1], [625.45], [286.61], [0.0], [512.0], [512.0], [512.0], [2048.0],
    [fH_freq1], [615.56], [332.67], [0.0], [512.0], [512.0], [512.0], [2048.0],
    [fL_freq2], [1122.74], [671.55], [0.0], [513.0], [1130.0], [1821.0], [2048.0],
    [fH_freq2], [1072.98], [772.97], [0.0], [214.5], [1164.0], [1801.5], [2048.0],
    
    // Topper (Kun 1 og 2)
    table.cell(colspan: 8, [*Spektrale topper (Magnitude)*]),
    [fL_peak1], [0.0382], [0.1477], [0.0004], [0.0005], [0.0006], [0.0006], [0.9372],
    [fH_peak1], [0.0011], [0.0030], [0.0004], [0.0005], [0.0005], [0.0006], [0.0313],
    [fL_peak2], [0.0340], [0.1309], [0.0003], [0.0003], [0.0004], [0.0004], [0.9180],
    [fH_peak2], [0.0009], [0.0028], [0.0003], [0.0004], [0.0004], [0.0004], [0.0270],
    
    // Aggregert statistikk
    table.cell(colspan: 8, [*Aggregerte statistikker (Amplitude/Energi)*]),
    [fL_max],   [0.0382], [0.1477], [0.0004], [0.0005], [0.0006], [0.0006], [0.9372],
    [fH_max],   [0.0011], [0.0030], [0.0004], [0.0005], [0.0005], [0.0006], [0.0313],
    [fL_mean],  [0.0035], [0.0127], [0.0001], [0.0001], [0.0001], [0.0001], [0.0813],
    [fH_mean],  [0.0002], [0.0004], [0.0001], [0.0001], [0.0001], [0.0001], [0.0040],
    [fL_std],   [0.0049], [0.0185], [0.0001], [0.0001], [0.0001], [0.0001], [0.1130],
    [fH_std],   [0.0001], [0.0004], [0.0000], [0.0001], [0.0001], [0.0001], [0.0043],
    [fL_min],   [0.0000], [0.0001], [0.0000], [0.0000], [0.0000], [0.0000], [0.0005],
    [fH_min],   [0.0000], [0.0000], [0.0000], [0.0000], [0.0000], [0.0000], [0.0000],
  )
) <feature_stats_filtered>

En nærmere analyse av dataene valgt ut for MLP gir viktig innsikt i signalenes natur og hvordan de bør behandles før de introduseres for MLP-modellen. Et sentralt teknisk poeng er forholdet mellom den valgte FFT-størrelsen og de resulterende frekvensindeksene. Selv om vi har definert en FFT-størrelse på 4096 i koden, ser vi at frekvensindeksene kun strekker seg opp til 2048. Dette er en direkte konsekvens av at vi benytter en transformasjon av reelle signaler (rFFT). Siden inngangssignalet ikke har en imaginær del, blir det positive og negative frekvensspekteret symmetrisk. Algoritmen returnerer derfor kun den første halvparten av spekteret, N/2, som representerer de unike frekvenskomponentene fra 0 opp til Nyquist-frekvensen. Dette er tilstrekkelig for vår analyse, da all relevant informasjon for å skille mellom dronemodusene er bevart i disse 2048 indeksene.

Når vi studerer statistikken for de ulike trekkene, ser vi en tydelig diktometri i dataene. Frekvensindeksene opererer på en lineær skala mellom 0 og 2048, mens de spektrale magnitudene (peaks) og statistiske målene som gjennomsnitt og standardavvik ofte har ekstremt lave verdier, gjerne i størrelsesorden $10^(−4)$ til $10^(−5)$. Den lave standardavviket i mange av de statistiske trekkene, spesielt for det høye båndet (H), indikerer at mye av dataene består av bakgrunnsstøy med svært liten varians. Samtidig ser vi i kolonnene for maksimalverdier at enkelte opptak har betydelig høyere utslag. Dette betyr at informasjonen modellen skal lære av er svært spredt altså at de viktige signalene fremstår som sjeldne topper i et ellers flatt og konsistent støygulv.

Denne enorme forskjellen i tallverdier, fra tusener i frekvensindekser til brøkdeler i magnitude, gjør det tvingende nødvendig å benytte en StandardScaler før MLP-trening. Uten en slik standardisering, der hver feature transformeres til å ha null i gjennomsnitt og en standardavvik på én, vil modellen i praksis ignorere de spektrale magnitudene og kun legge vekt på frekvensindeksene fordi disse har numerisk størst påvirkning på vektene i nettverket. Ved å skalere dataene sikrer vi at de subtile forskjellene i signalstyrke og støygulv får like stor betydning i beslutningsprosessen som de dominerende frekvensene.

Til slutt viser den statistiske oversikten at det finnes redundans i det nåværende settet av features som med fordel kan fjernes for å effektivisere modellen. For eksempel ser vi at fL_max og fL_peak1 er identiske (og tilsvarende for H-båndet) i alle statistiske mål, noe som er logisk siden den største spektrale toppen per definisjon også er signalets maksimalverdi. Å beholde begge bidrar ikke med ny informasjon, men øker dimensjonaliteten unødvendig.

Videre viser analysen at features som fL_min og fH_min har et ekstremt lav standardavvik og varians. Dette indikerer at verdiene er tilnærmet konstante på tvers av alle opptakene, og dermed ikke inneholder diskriminerende informasjon som modellen kan bruke til å skille mellom de ulike dronene eller modusene. At disse verdiene er så flate, betyr at de i praksis kun representerer et statisk støygulv.

Dette var sammenhenger som ikke ble innsett i den initielle utvelgelsen av features, men som ble åpenbare gjennom EDA, noe som understreker nytten av en slik statistisk gjennomgang. En seleksjonsprosess hvor vi fjerner duplikater som maksimalverdier, samt filtrerer ut trekk med aller lavest varians som minimumsverdier, vil gjøre MLP-modellen mer robust mot overfitting og raskere å trene, ettersom den får et renere og mer destillert bilde av de faktiske RF-signaturene.


== Train/test-splitt
Datasettet ble delt i trenings- og testdata før modelltrening. Vi brukte en stratified train/test-splitt, slik at klassefordelingen i størst mulig grad ble bevart i både trenings- og testsettet. Dette er spesielt viktig ved modusklassifisering, siden klassene kan være ubalanserte.

En viktig utfordring med DroneRF-datasettet er risiko for data leakage. Siden lange opptak deles inn i mange mindre vinduer, kan svært like segmenter ende opp i både trenings- og testsett dersom splitten gjøres etter vinduing. Dette kan gi kunstig høy ytelse, fordi modellen da testes på signaler som ligner sterkt på signaler den allerede har sett under trening. For å redusere denne risikoen bør splitten ideelt sett gjøres på opptaks- eller filnivå før vinduene genereres. Dersom dette ikke er fullstendig implementert, må det nevnes som en begrensning i prosjektet.

I vår implementasjon bruker vi en fast random seed for å gjøre splitten reproduserbar. Teststørrelsen er satt til 20 %, mens resten brukes til trening. For modellene brukes testsettet også som valideringsgrunnlag under trening, blant annet for å lagre beste modell basert på valideringsytelse. Dette diskuteres videre som en metodisk begrensning, siden et separat valideringssett ville gitt en renere evaluering.


== Forbehandling

Før modelltrening måtte RF-dataene forbehandles. Først ble råfilene lest inn og koblet til riktige labels basert på BUI-koden i filnavnet. Deretter ble signalene delt inn i mindre vinduer. For MLP-modellen ble hvert vindu omgjort til en feature-vektor. Disse feature-vektorene ble standardisert før trening, slik at features med stor numerisk skala ikke skulle dominere læringen.

For CNN-modellen ble råsignalet bevart i større grad. Lavt og høyt frekvensbånd ble representert som to separate kanaler, slik at modellen kunne lære mønstre på tvers av begge mottakerbåndene. Signalene ble tilpasset en fast sekvenslengde ved å kutte eller fylle med nuller der det var nødvendig. I tillegg ble signalene normalisert basert på statistikk fra treningssettet, slik at modellen fikk mer stabile inputverdier.

Denne forbehandlingen gjør datasettet egnet for to ulike modelltyper: MLP, som lærer fra konstruerte feature-vektorer, og CNN, som lærer direkte fra strukturen i råsignalet.
