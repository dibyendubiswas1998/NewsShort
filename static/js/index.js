const hamburger = document.querySelector('.hamburger');
const navLink = document.querySelector('.nav__link');

hamburger.addEventListener('click', () => {
  navLink.classList.toggle('hide');
});

const nav = document.querySelector('.nav');
const section = document.querySelector('section');

window.addEventListener('scroll', () => {
  if (window.scrollY > section.offsetTop) {
    nav.classList.add('fixed');
  } else {
    nav.classList.remove('fixed');
  }
});

