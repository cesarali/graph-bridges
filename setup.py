import os
import re
import subprocess

import setuptools


def load_requirements():
    try:
        with open("requirements.txt") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("WARNING: requirements.txt not found")
        return []


try:
    with open("README.md", "r") as f:
        long_description = f.read()
except:
    long_description = "# discrete_diffusion"

# make version string
git_describe = "git describe --tags --always --dirty"
if os.getenv("VERSION") is not None and os.environ.get("VERSION") != "":
    version = os.environ["VERSION"]
elif (git_description := subprocess.run(git_describe, capture_output=True, shell=True)).returncode == 0:
    semantic_versioning = re.compile("v?(\d+(\.\d+(\.\d+)?)?)")
    semantic_versioning_post = re.compile("v?(\d+(\.\d+(\.\d+)?)?)-(\d+)-(\w+)")
    semantic_versioning_post_dirty = re.compile("v?(\d+(\.\d+(\.\d+)?)?)-(\d+)-(\w+)-dirty")
    semantic_versioning_dirty = re.compile("v?(\d+(\.\d+(\.\d+)?)?)-dirty")
    pure_commit_dirty = re.compile("(\w+)-dirty")

    git_description.stdout = git_description.stdout.decode("utf-8").strip()

    if match := semantic_versioning.fullmatch(git_description.stdout):
        version = match.group(1)
    elif match := semantic_versioning_post.fullmatch(git_description.stdout):
        version = match.group(1) + ".post" + match.group(4) + "+commit" + match.group(5)
    elif match := semantic_versioning_post_dirty.fullmatch(git_description.stdout):
        version = match.group(1) + ".post" + match.group(4) + "+commit" + match.group(5) + ".dirty"
    elif match := semantic_versioning_dirty.fullmatch(git_description.stdout):
        version = match.group(1) + "+dirty"
    elif match := pure_commit_dirty.fullmatch(git_description.stdout):
        version = "0.0.0+commit" + match.group(1) + ".dirty"
    else:
        version = "0.0.0+commit" + git_description.stdout

setuptools.setup(
    name="discrete_diffusion",
    version=version if version else "0.0.0+unknownversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:cesarali/discrete_diffusion.git",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[],
    python_requires=">=3.10",
    install_requires=load_requirements(),
    index_url=["https://download.pytorch.org/whl/cu118"],
    include_package_data=True,
)
