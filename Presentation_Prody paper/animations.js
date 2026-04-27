
const slides = [
    "1-title.html",
    "2-Outline.html",
    "3-Motivation_problem.html",
    "4-existingapproaches.html",
    "5-Nesy_solution.html",
    "6-Framework.html",
    "7-prefix_extraction.html",
    "8-encodings.html",
    "9-one_hot_encoding.html",
    "10-Indexbased_encoding.html",
    "11-ShrinkedIndex_encoding.html",
    "12-multi_encoders.html",
    "13-training.html",
    "14-predictionlayer.html",
    "15-BS_details.html",
    "16-predictionlayer_step3.html",
    "17-experimentsetup.html",
    "18-datasets.html",
    "19-activity_prediction_results.html",
    "20-act-res_prediction_results.html",
    "21-Conclusions.html",
    "thanks.html"
];


document.addEventListener("DOMContentLoaded", () => {
  const slide = document.querySelector("[slide]");
  if (slide) {
    document.querySelector(".slide-number").textContent =
      parseInt(slide.getAttribute("slide"));
  }
});



let currentStep = 0;

// Collect step elements sorted by data-order
const steps = [...document.querySelectorAll("[data-order]")]
              .sort((a, b) => a.dataset.order - b.dataset.order);

// Get current slide index from the [slide] attribute
const currentSlideIndex = parseInt(document.querySelector("[slide]").getAttribute("slide")) ;




function showNextElement() {
    if (currentStep < steps.length) {
        const order = steps[currentStep].dataset.order;
        const img1 = document.getElementById(`img${order-1}`);
        const img2 = document.getElementById(`img${order}`);
        if (img1 && img2) {
             img1.replaceWith(img2);
             img2.style.display = 'block';
         }
        document.querySelectorAll(`[data-order='${order}']`)
                .forEach(el => {
                    el.classList.remove("hidden");
                    el.classList.add("visible");
                });


        while (currentStep < steps.length && steps[currentStep].dataset.order === order) {
            currentStep++;
        }
    } else {
        if (currentSlideIndex < slides.length -1) {
            window.location.href = slides[currentSlideIndex + 1];
        }
        else{
            window.location.href = slides[0];
        }
    }
}

document.addEventListener("click", showNextElement);

document.addEventListener("keydown", function(e) {
    if ((e.key === "Enter") || (e.key === "ArrowRight")) {
        showNextElement();
    }
    if (e.key === "ArrowLeft" && currentSlideIndex > 0) {
        window.location.href = slides[currentSlideIndex - 1];
    }
});

