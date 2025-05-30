$(document).ready(function () {
  $("#uploadBtn").on("click", function (event) {
    event.preventDefault();
    var formData = new FormData(document.getElementById("uploadForm"));

    // Get the video URL input
    var videoUrl = $("#videoUrl").val();
    formData.append("videoUrl", videoUrl);

    $("#stableFrames").attr("src", "");
    $("#yoloFrames").attr("src", "");
    $("#fireAnalysis").attr("src", "");

    $.ajax({
      url: "/upload",
      type: "POST",
      data: formData,
      contentType: false,
      processData: false,
      success: function (response) {
        if (response.error) {
          $("#logs").innerText = response.error;
        } else {
          $("#logs").innerText = response.result;
        }
        // Is there any better way of refreshing the socket connection!
        // setTimeout(function () {
        //   location.reload();
        // }, 2000);
      },
      error: function () {
        $("#logs").innerText = "Error uploading video. Please try again.";

        // Is there any better way of refreshing the socket connection!
        // setTimeout(function () {
        //   location.reload();
        // }, 2000);
      },
    });
  });

  var socket = io.connect();

  $("#stopBtn").on("click", function (event) {
    event.preventDefault();
    socket.emit("stop_processing");

    // Is there any better way of refreshing the socket connection!
    // setTimeout(function () {
    //   location.reload();
    // }, 2000);
  });

  socket.on("stable_update", function (data) {
    $("#stableFrames").attr("src", "data:image/png;base64," + data);
  });

  socket.on("yolo_update", function (data) {
    $("#yoloFrames").attr("src", "data:image/png;base64," + data);
  });

  socket.on("analysis", function (data) {
    $("#fireAnalysis").attr("src", "data:image/png;base64," + data);
  });
});