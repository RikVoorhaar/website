// Select a random background image for the page hero every time the page is loaded.

// Path: assets/js/usr/random-background.js

// const fs = require("fs");
const heroBackground = document.querySelector(".page__hero-background");
const body = document.querySelector("html");
const imagePath = "/assets/images/banners/compressed";

let images;
(async () => {
  const loadImages = async () => {
    try {
      const response = await fetch("/assets/images/backgrounds.json");
      const data = await response.json();
      return data.images;
    } catch (error) {
      console.error(error);
    }
  };

  images = await loadImages();
  setRandomBackground();
})();

function setRandomBackground() {
  const index = Math.floor(Math.random() * images.length);
  const image = images[index];
  const imageURL = `url("${imagePath}/${image}")`;
  const heroBackgroundStyle = `linear-gradient(rgba(255,255,255,0.5), rgba(255,255,255,0)), ${imageURL}`;
  heroBackground.style.setProperty("background-image", heroBackgroundStyle);
  const mainBackgroundStyle = `linear-gradient(rgba(255,255,255,0.95), rgba(255,255,255,0.85)), ${imageURL}`;
  body.style.setProperty("background-image", mainBackgroundStyle);
}

