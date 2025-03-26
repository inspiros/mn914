## Developer Notes

#### How to purge file from history

Do not ever commit a large file (e.g. model checkpoint) to the repository.
Use this command to undo if you have accidentally done it:

```cmd
git filter-branch -f --tree-filter 'rm -f <path_to_file>' HEAD
git push origin --force --all
```
