$(document).ready(function () {
  window.sortTable = function (columnIndex) {
    var table = document.getElementById("modelTable");
    var rows = Array.from(table.rows).slice(1);
    var sortedRows = rows.sort((a, b) => {
      var cellA = parseFloat(a.cells[columnIndex].innerText) || 0;
      var cellB = parseFloat(b.cells[columnIndex].innerText) || 0;
      return cellB - cellA;
    });
    sortedRows.forEach((row) => table.appendChild(row));
  };

  $.getJSON(modelDataUrl, function (data) {
    var tableBody = $("#modelTable tbody");
    data.forEach(function (item) {
      var row =
        "<tr>" +
        "<td>" +
        item.model +
        "</td>" +
        "<td>" +
        item.precision +
        "</td>" +
        "<td>" +
        item.recall +
        "</td>" +
        "<td>" +
        item.map50 +
        "</td>" +
        "<td>" +
        item.map50_95 +
        "</td>" +
        "<td>" +
        item.iou +
        "</td>" +
        "<td>" +
        item.epochs +
        "</td>" +
        "</tr>";
      tableBody.append(row);
    });
  });
});
