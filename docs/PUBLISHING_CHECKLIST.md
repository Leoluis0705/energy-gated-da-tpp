# GitHub and Zenodo publishing checklist

1. Create an empty GitHub repository: do not initialize it with a README,
   `.gitignore`, or license because those files are maintained locally.
2. Confirm the GitHub owner and repository name.
3. Confirm public visibility; Zenodo's GitHub release integration requires a
   public repository.
4. Obtain author/institution approval for a code license and a data license.
   Select **No license** on the GitHub creation form until that approval exists.
5. Renew GitHub CLI authentication.
6. Run `python scripts/validate_repository.py --tests --figures --manifest`.
7. Review `git status` and the SHA-256 manifest.
8. Push the verified commit and create a `v1.0.0` GitHub release.
9. Enable the repository in Zenodo and archive the release.
10. Record the Zenodo DOI and GitHub release URL.
11. Add the real DOI/URL to `CITATION.cff`, `.zenodo.json`, the manuscript Data
    Availability statement, and the reference list.

Recommended licenses, subject to author/institution approval:

- code: MIT;
- original tabular data and figures: CC BY 4.0;
- third-party datasets and model weights: retain their upstream terms and do
  not relicense them.
