import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Brand fonts from bebeautifulhair.com, bundled so they load fast and work offline.
import "@fontsource/reem-kufi/400.css";
import "@fontsource/reem-kufi/500.css";
import "@fontsource/reem-kufi/600.css";
import "@fontsource/reem-kufi/700.css";
import "@fontsource/young-serif/400.css";
import "./index.css";
import CustomerApp from "./customer/CustomerApp.jsx";
import AdminApp from "./admin/AdminApp.jsx";

const isAdmin = window.location.pathname.startsWith("/admin");

createRoot(document.getElementById("root")).render(
  <StrictMode>{isAdmin ? <AdminApp /> : <CustomerApp />}</StrictMode>
);
