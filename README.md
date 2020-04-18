# Hvordan organisere outputs
Når du kjører en trening du vil ta vare på: finn på et navn som beskriver det eksperimentet, lagre configfila du brukte som `configs/navn.yaml` og sett `OUTPUT_DIR` til det samme navnet, f.eks `configs/resnet50_pretrained_very_deep.yaml` og `outputs/resnet50_pretrained_very_deep`.

Commit configfila sammen med/etter endringene i koden som du kjørte med, sånn at commiten som legger til denne configfila vil ha den versjonen av koden som du brukte til å trene. Kan evt. også gjøre `git tag <navn>` for å gjøre det lettere å finne senere.

Når treningen er ferdig, last ned eller commit `tf_logs` fila som ligger under `outputs/navn`, eller hele `output/navn` (men ikke checkpointene .pth) Kan f.eks laste ned med rsync:

`rsync -rP  --exclude='*.pth' BRUKERNAVN@oppdal.idi.ntnu.no:path/to/outputs/ experiments`

så kan vi samle det sammen i en felles mappe som man kan kjøre tensorboard på lokalt.

# Tensorboard

ssh -L 127.0.0.1:1234:127.0.0.1:6006 BRUKERNAVN@oppdal.idi.ntnu.no

tensorboard --logdir outputs



 http://localhost:1234/#scalars
