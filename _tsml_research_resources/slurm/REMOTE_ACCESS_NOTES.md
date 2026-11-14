# Notes for Claude on Tony's laptop: ssh access to Iridis and pulling results

Written 2026-11-06 on the Windows desktop, where the same setup was done for Hali.
The aim: Claude runs non-interactive commands on Iridis over ssh, and pulls results
from Iridis to the laptop, so they can be compared there. Read this, then
`README.md` and `HYDRA_NOTES.md` in this directory.

## Hosts and accounts

From Tony's saved PuTTY sessions:

| Cluster | Login | VPN |
|---|---|---|
| Iridis 6 (CPU) | `ajb2u23@iridis6_b.soton.ac.uk` (the name as saved in PuTTY) | Southampton |
| Iridis X (GPU) | `ajb2u23@loginX001.iridis.soton.ac.uk` or `loginX003` | Southampton |
| Hali (UEA) | `ajb@hali.uea.ac.uk` | UEA |

- Both clusters need their university's VPN. One VPN at a time in practice, so Iridis
  and Hali are not reachable together, and a machine on the Southampton VPN cannot
  talk to Hali (or the other way round).
- Iridis 6 and Iridis X share one home filesystem: results written on either are
  under the same `~/Results/EEG`, so pull from one of them only.
- If a host name does not resolve, ask Tony for the address he uses in PuTTY.

## One-time key setup (Tony does the password step)

Tony cannot paste into Git Bash, and Claude cannot type passwords. So:

1. Claude makes a key with no passphrase, if there is none (no prompt needed):
   `[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -C "tony-laptop" -f ~/.ssh/id_ed25519`
2. Claude writes a short script in Tony's home, e.g. `~/iridis_key.sh`, that
   installs the key, so Tony types only `bash iridis_key.sh` in Git Bash
   (Windows key, type "Git Bash"; it starts in the home folder):

   ```bash
   H=ajb2u23@iridis6_b.soton.ac.uk
   ssh -o StrictHostKeyChecking=accept-new "$H" \
     "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" \
     < ~/.ssh/id_ed25519.pub && ssh -o BatchMode=yes "$H" hostname
   ```

   Tony types his password once, invisibly. One install covers Iridis 6 and X
   (shared home).
3. Claude tests: `ssh -o BatchMode=yes ajb2u23@<host> hostname` must print a host
   name with no prompt.

If the key is offered but refused (`ssh -v` shows "Offering public key" then no
more methods), check permissions on the cluster, which Tony can run in PuTTY:
`ls -ld ~ ~/.ssh ~/.ssh/authorized_keys`. sshd ignores keys if the home folder is
group- or world-writable: Hali's home was `drwxrwsrwx`, and `chmod go-w ~` fixed
it. `~/.ssh` must be `drwx------` and `authorized_keys` `-rw-------`: on Iridis
`~/.ssh` was `drwxrwxrwx`, fixed with `chmod 700 ~/.ssh`. If the password step of
`iridis_key.sh` fails, paste the `echo '<pubkey>' >> ~/.ssh/authorized_keys` line
in PuTTY instead (right-click pastes there).

To revoke: delete the `tony-laptop` line from `~/.ssh/authorized_keys` on Iridis.

## Using the connection

- Hali logins took about 40 seconds each, and Iridis may be slow too: batch
  commands into one `ssh` call, don't make many small ones.
- Always `-o BatchMode=yes`: a prompt fails fast instead of hanging.
- Commands longer than ~2 minutes: run in the background and wait for the
  notification.
- A login lands on one login node of several; screen sessions (Iridis X supervisor)
  live on one node, so `screen -ls` may show nothing from another node.
- Read-only is fine without asking: `squeue -u $USER`, `sacct`, `python
  tsml_slurm.py status <toml>`, reading logs, `ls`, `du`. **Ask Tony first** before
  anything that changes the cluster: submitting or cancelling jobs, `start`/`stop`,
  `retry`, deleting or moving results, changing envs.

## Pulling results

`pull_results.sh` (this directory) pulls new or changed files only, without rsync,
in two logins: it lists the remote files with sizes, compares with the local copy,
and streams the missing ones as a gzip tar. It never deletes or changes anything.
Tested against Iridis 6 on 2026-11-06 from the laptop: 1622 Predictions files
(165 MB) in about a minute, and a rerun fetched nothing. The laptop copy is in
`C:/Results/EEG`. The default pattern also takes `_superseded/` and `NBack/`.

```
bash pull_results.sh ajb2u23@iridis6_b.soton.ac.uk Results/EEG <local results dir>
bash pull_results.sh ajb2u23@iridis6_b.soton.ac.uk Results/EEG <local dir> "./DistanceBased/RiemannianMDM-OAS/Predictions/*"
```

- The default pattern takes `Predictions` only. Skip `Checkpoints`: they are large
  (they made an earlier results zip 8 GB) and are deleted when a task finishes.
- Results layout: `<Category>/<Estimator>/Predictions/<Dataset>/{train,test}Resample0.csv`.
- The master results copy is meant to live on Hydra; ask Tony where the laptop
  copy should go and whether it should merge with anything.
- Analysis scripts used before (league table, head-to-head, HC2 from component
  files): `build_hc2_from_files.py` here; the league and head-to-head scripts were
  scratch scripts on the desktop, so ask Tony if he wants them recreated.
