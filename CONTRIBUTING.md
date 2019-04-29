# Instructions for Contributors

To contribute to this project, you should first create your own fork, and remember to periodically [sync changes from this repository](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository). You can then create [pull requests](https://yangsu.github.io/pull-request-tutorial/) for modifications you have made. Your changes will be tested and reviewed before they are merged into this repository. If you are not familiar with forks and pull requests, we recommend doing a Google or YouTube search to find many useful tutorials on the topic.

Also, you can have a look at the [codebase structure](docs/codebase.md) before getting started.

A suggested flow for contributing would be:
First, open up a new feature branch to solve an existing bug/issue
```bash
$ git checkout -b <feature-branch> upstream/master
```
This ensures that the branch is up-to-date with the `master` branch of the main repository, irrespective of the status of your forked repository.

Once you are done making commits of your changes / adding the feature, you can:
(In case this is the first set of commits from this _new_ local branch)
```bash
git push --set-upstream origin 
```
(Assuming the name of your forked repository remote is `origin`), which will create a new branch `<feature-branch>`
tracking your local `<feature-branch>`, in case it hasn't been created already.

Then, create a [pull request](https://help.github.com/en/articles/about-pull-requests) in this repository.