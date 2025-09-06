// DOM Elements
const uploadArea = document.getElementById("uploadArea");
const videoInput = document.getElementById("videoInput");
const youtubeInput = document.getElementById("youtubeInput");
const youtubeUrl = document.getElementById("youtubeUrl");
const processYoutubeBtn = document.getElementById("processYoutubeBtn");
const processingStatus = document.getElementById("processingStatus");
const progressFill = document.getElementById("progressFill");
const statusText = document.getElementById("statusText");
const results = document.getElementById("results");
const navLinks = document.querySelectorAll(".nav-link");
const methodToggles = document.querySelectorAll(".method-toggle");

// Smooth scrolling for navigation
function scrollToSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (section) {
    section.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }
}

// Update active navigation link
function updateActiveNavLink() {
  const sections = document.querySelectorAll("section[id]");
  const scrollPos = window.scrollY + 100;

  sections.forEach((section) => {
    const sectionTop = section.offsetTop;
    const sectionHeight = section.offsetHeight;
    const sectionId = section.getAttribute("id");
    const navLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);

    if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
      navLinks.forEach((link) => link.classList.remove("active"));
      if (navLink) {
        navLink.classList.add("active");
      }
    }
  });
}

// YouTube URL validation
function validateYouTubeUrl(url) {
  const youtubeRegex =
    /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[\w-]+/;
  return youtubeRegex.test(url);
}

// Extract YouTube video ID from URL
function extractYouTubeId(url) {
  const regex = /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/;
  const match = url.match(regex);
  return match ? match[1] : null;
}

// File upload handling
function handleFileUpload(file) {
  if (!file) return;

  // Validate file type
  const validTypes = [
    "video/mp4",
    "video/mov",
    "video/avi",
    "video/mkv",
    "video/webm",
  ];
  if (!validTypes.includes(file.type)) {
    alert("Please upload a valid video file (MP4, MOV, AVI, MKV, or WebM)");
    return;
  }

  // Validate file size (max 500MB)
  const maxSize = 500 * 1024 * 1024; // 500MB
  if (file.size > maxSize) {
    alert("File size must be less than 500MB");
    return;
  }

  // Start processing simulation
  startProcessing(file, "file");
}

// YouTube URL handling
function handleYouTubeUrl(url) {
  if (!url.trim()) {
    alert("Please enter a YouTube URL");
    return;
  }

  if (!validateYouTubeUrl(url)) {
    alert(
      "Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)"
    );
    return;
  }

  const videoId = extractYouTubeId(url);
  if (!videoId) {
    alert("Could not extract video ID from URL");
    return;
  }

  // Start processing simulation with YouTube data
  const youtubeData = {
    url: url,
    videoId: videoId,
    title: "YouTube Video", // In real app, this would be fetched from YouTube API
    duration: "12:34", // In real app, this would be fetched from YouTube API
    size: "Unknown", // YouTube videos don't have a fixed size
  };

  startProcessing(youtubeData, "youtube");
}

// Simulate video processing
function startProcessing(data, type) {
  // Hide upload areas and show processing status
  uploadArea.style.display = "none";
  youtubeInput.style.display = "none";
  processingStatus.style.display = "block";
  results.style.display = "none";

  // Different processing steps based on type
  let processingSteps;
  if (type === "youtube") {
    processingSteps = [
      { text: "Downloading YouTube video...", progress: 15 },
      { text: "Analyzing video content...", progress: 35 },
      { text: "Identifying key moments...", progress: 55 },
      { text: "Extracting important segments...", progress: 75 },
      { text: "Generating summary video...", progress: 90 },
      { text: "Finalizing output...", progress: 100 },
    ];
  } else {
    processingSteps = [
      { text: "Analyzing video content...", progress: 20 },
      { text: "Identifying key moments...", progress: 40 },
      { text: "Extracting important segments...", progress: 60 },
      { text: "Generating summary video...", progress: 80 },
      { text: "Finalizing output...", progress: 100 },
    ];
  }

  let currentStep = 0;

  const processStep = () => {
    if (currentStep < processingSteps.length) {
      const step = processingSteps[currentStep];
      statusText.textContent = step.text;
      progressFill.style.width = `${step.progress}%`;
      currentStep++;
      setTimeout(processStep, 1500);
    } else {
      // Processing complete
      setTimeout(() => {
        showResults(data, type);
      }, 1000);
    }
  };

  processStep();
}

// Show processing results
function showResults(data, type) {
  processingStatus.style.display = "none";
  results.style.display = "block";

  // Update file information
  const originalDuration = document.querySelector(".video-item .duration");
  const originalSize = document.querySelector(".video-item .size");
  const summaryDuration = document.querySelector(
    ".video-item:nth-child(3) .duration"
  );
  const summarySize = document.querySelector(".video-item:nth-child(3) .size");

  // Simulate file info based on type
  let simulatedDuration,
    originalSizeValue,
    summaryDurationValue,
    summarySizeValue;

  if (type === "youtube") {
    simulatedDuration = data.duration || "12:34";
    originalSizeValue = "YouTube Video";
    summaryDurationValue = "2:15"; // Simulated summary duration
    summarySizeValue = "45 MB"; // Simulated summary size
  } else {
    const fileSizeMB = (data.size / (1024 * 1024)).toFixed(1);
    simulatedDuration = "15:32"; // This would be extracted from the video
    originalSizeValue = `${fileSizeMB} MB`;
    summaryDurationValue = "3:45"; // This would be calculated
    summarySizeValue = `${(fileSizeMB * 0.25).toFixed(1)} MB`; // Simulate 75% reduction
  }

  originalDuration.textContent = simulatedDuration;
  originalSize.textContent = originalSizeValue;
  summaryDuration.textContent = summaryDurationValue;
  summarySize.textContent = summarySizeValue;

  // Scroll to results
  setTimeout(() => {
    results.scrollIntoView({ behavior: "smooth", block: "center" });
  }, 500);
}

// Reset to upload state
function resetToUpload() {
  uploadArea.style.display = "block";
  youtubeInput.style.display = "none";
  processingStatus.style.display = "none";
  results.style.display = "none";
  progressFill.style.width = "0%";
  videoInput.value = "";
  youtubeUrl.value = "";

  // Reset method toggle to file upload
  methodToggles.forEach((toggle) => toggle.classList.remove("active"));
  methodToggles[0].classList.add("active");
}

// Toggle between upload methods
function toggleUploadMethod(method) {
  methodToggles.forEach((toggle) => toggle.classList.remove("active"));
  document.querySelector(`[data-method="${method}"]`).classList.add("active");

  if (method === "file") {
    uploadArea.style.display = "block";
    youtubeInput.style.display = "none";
  } else {
    uploadArea.style.display = "none";
    youtubeInput.style.display = "block";
  }
}

// Video cycling functionality
function cycleHeroVideos() {
  const video = document.getElementById("heroVideo");
  if (!video) return;

  const videoSources = [
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
  ];

  let currentIndex = 0;

  function switchVideo() {
    // Fade out current video
    video.style.opacity = "0";

    // After fade out completes, switch video
    setTimeout(() => {
      video.src = videoSources[currentIndex];
      video.load();
      currentIndex = (currentIndex + 1) % videoSources.length;

      // Fade in new video
      video.style.opacity = "1";
    }, 300); // 300ms fade transition
  }

  // Handle when video ends - switch to next video
  function handleVideoEnded() {
    switchVideo();
  }

  // Hide play button when video is playing
  function handleVideoPlay() {
    const overlay = document.querySelector(".video-overlay");
    if (overlay) {
      overlay.style.display = "none";
    }
  }

  // Show play button when video is paused
  function handleVideoPause() {
    const overlay = document.querySelector(".video-overlay");
    if (overlay) {
      overlay.style.display = "flex";
    }
  }

  // Add event listeners for video play/pause
  video.addEventListener("play", handleVideoPlay);
  video.addEventListener("pause", handleVideoPause);
  video.addEventListener("ended", handleVideoEnded);

  // Switch video every 15 seconds (increased from 8)
  setInterval(switchVideo, 15000);

  // Initial video load
  video.src = videoSources[0];
  video.load();
}

// Theme Management
class ThemeManager {
  constructor() {
    this.currentTheme = localStorage.getItem("theme") || "light";
    this.themes = ["light", "dark", "aurora"];
    this.themeIndex = this.themes.indexOf(this.currentTheme);
    this.init();
  }

  init() {
    this.applyTheme(this.currentTheme);
    this.updateThemeButton();
  }

  applyTheme(theme) {
    console.log("Applying theme:", theme);
    document.documentElement.setAttribute("data-theme", theme);
    this.currentTheme = theme;
    localStorage.setItem("theme", theme);

    // Force immediate style update
    document.body.style.backgroundColor =
      theme === "dark" ? "#000000" : theme === "aurora" ? "#0a0a0a" : "#ffffff";
    document.body.style.color = theme === "light" ? "#000000" : "#ffffff";

    // Verify the attribute was set
    console.log(
      "Document element data-theme:",
      document.documentElement.getAttribute("data-theme")
    );
    console.log(
      "Computed background color:",
      getComputedStyle(document.body).backgroundColor
    );

    // Handle solar flares
    if (solarFlaresManager && solarFlaresManager.flares) {
      if (theme === "aurora") {
        solarFlaresManager.flares.forEach((flare) => {
          flare.style.display = "block";
        });
      } else {
        solarFlaresManager.flares.forEach((flare) => {
          flare.style.display = "none";
        });
        document.body.classList.remove("scrolling");
      }
    }
  }

  nextTheme() {
    this.themeIndex = (this.themeIndex + 1) % this.themes.length;
    const newTheme = this.themes[this.themeIndex];
    this.applyTheme(newTheme);
    this.updateThemeButton();
  }

  updateThemeButton() {
    const themeBtn = document.getElementById("themeToggle");
    const icon = themeBtn.querySelector("i");

    switch (this.currentTheme) {
      case "light":
        icon.className = "fas fa-sun";
        themeBtn.title = "Switch to Dark Mode";
        break;
      case "dark":
        icon.className = "fas fa-moon";
        themeBtn.title = "Switch to Aurora Mode";
        break;
      case "aurora":
        icon.className = "fas fa-magic";
        themeBtn.title = "Switch to Light Mode";
        break;
    }
  }
}

// Solar Flares Manager
class SolarFlaresManager {
  constructor() {
    this.flares = [];
    this.isActive = false;
    this.scrollTimeout = null;
    this.init();
  }

  init() {
    this.createFlares();
    this.setupScrollListener();
  }

  createFlares() {
    // Create 25 solar flare particles for more prominent effect
    for (let i = 0; i < 25; i++) {
      const flare = document.createElement("div");
      flare.className = "solar-flare";

      // Random positioning
      flare.style.left = Math.random() * 100 + "%";
      flare.style.top = "100vh";

      // Random size variation (larger particles)
      const size = Math.random() * 4 + 4; // 4-8px
      flare.style.width = size + "px";
      flare.style.height = size + "px";

      // Random animation delay
      flare.style.animationDelay = Math.random() * 3 + "s";

      // Initially hidden
      flare.style.display = "none";

      document.body.appendChild(flare);
      this.flares.push(flare);
    }
  }

  setupScrollListener() {
    let isScrolling = false;
    let scrollTimer = null;

    window.addEventListener("scroll", () => {
      if (!isScrolling) {
        isScrolling = true;
        console.log("Scroll started, activating flares");
        this.activateFlares();
      }

      // Clear existing timer
      if (scrollTimer) {
        clearTimeout(scrollTimer);
      }

      // Set new timer
      scrollTimer = setTimeout(() => {
        isScrolling = false;
        console.log("Scroll stopped, deactivating flares");
        this.deactivateFlares();
      }, 200);
    });
  }

  activateFlares() {
    if (themeManager.currentTheme === "aurora") {
      this.isActive = true;
      document.body.classList.add("scrolling");

      // Show and restart animations for all flares
      this.flares.forEach((flare) => {
        flare.style.display = "block";
        flare.style.animationPlayState = "running";
        // Force reflow to restart animation
        flare.style.animation = "none";
        flare.offsetHeight; // Trigger reflow
        flare.style.animation = null;
      });
    }
  }

  deactivateFlares() {
    if (this.isActive) {
      this.isActive = false;
      document.body.classList.remove("scrolling");

      // Hide flares after a short delay
      setTimeout(() => {
        if (!this.isActive) {
          this.flares.forEach((flare) => {
            flare.style.display = "none";
            flare.style.animationPlayState = "paused";
          });
        }
      }, 100);
    }
  }

  destroy() {
    this.flares.forEach((flare) => {
      if (flare.parentNode) {
        flare.parentNode.removeChild(flare);
      }
    });
    this.flares = [];
  }
}

// Initialize managers
let themeManager;
let solarFlaresManager;

// Event Listeners
document.addEventListener("DOMContentLoaded", function () {
  // Initialize managers
  themeManager = new ThemeManager();
  solarFlaresManager = new SolarFlaresManager();

  console.log("Managers initialized");
  console.log("Initial theme:", themeManager.currentTheme);

  // Initialize video cycling
  cycleHeroVideos();

  // Theme toggle handler
  document.getElementById("themeToggle").addEventListener("click", () => {
    console.log(
      "Theme toggle clicked, current theme:",
      themeManager.currentTheme
    );
    themeManager.nextTheme();
    console.log("New theme:", themeManager.currentTheme);
  });
  // Upload area click handler
  uploadArea.addEventListener("click", () => {
    videoInput.click();
  });

  // File input change handler
  videoInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    handleFileUpload(file);
  });

  // Drag and drop handlers
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  });

  // Method toggle handlers
  methodToggles.forEach((toggle) => {
    toggle.addEventListener("click", () => {
      const method = toggle.getAttribute("data-method");
      toggleUploadMethod(method);
    });
  });

  // YouTube URL input handlers
  youtubeUrl.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      handleYouTubeUrl(youtubeUrl.value);
    }
  });

  processYoutubeBtn.addEventListener("click", () => {
    handleYouTubeUrl(youtubeUrl.value);
  });

  // Navigation link handlers
  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("href").substring(1);
      scrollToSection(targetId);
    });
  });

  // Scroll handler for navigation
  window.addEventListener("scroll", updateActiveNavLink);

  // Download button handler
  document.addEventListener("click", (e) => {
    if (
      e.target.closest(".btn-primary") &&
      e.target.closest(".results-actions")
    ) {
      // In a real app, this would trigger the actual download
      alert(
        "Download functionality would be implemented here. The processed video would be downloaded."
      );
    }
  });

  // Preview button handler
  document.addEventListener("click", (e) => {
    if (
      e.target.closest(".btn-secondary") &&
      e.target.closest(".results-actions")
    ) {
      // In a real app, this would open a video preview modal
      alert(
        "Preview functionality would be implemented here. A modal with the processed video would open."
      );
    }
  });

  // Add reset functionality (for demo purposes)
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && results.style.display !== "none") {
      resetToUpload();
    }
  });
});

// Intersection Observer for animations
const observerOptions = {
  threshold: 0.1,
  rootMargin: "0px 0px -50px 0px",
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = "1";
      entry.target.style.transform = "translateY(0)";
    }
  });
}, observerOptions);

// Observe mission cards for animation
document.addEventListener("DOMContentLoaded", () => {
  const missionCards = document.querySelectorAll(".mission-card");
  missionCards.forEach((card, index) => {
    card.style.opacity = "0";
    card.style.transform = "translateY(30px)";
    card.style.transition = `opacity 0.6s ease ${
      index * 0.1
    }s, transform 0.6s ease ${index * 0.1}s`;
    observer.observe(card);
  });
});

// Utility function to format file size
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// Utility function to format duration
function formatDuration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  } else {
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  }
}
