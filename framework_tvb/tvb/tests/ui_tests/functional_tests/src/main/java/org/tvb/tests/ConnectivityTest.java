package org.tvb.tests;

import org.junit.After;
import org.junit.Before;
import org.openqa.selenium.By;
import org.openqa.selenium.WebElement;

import java.util.List;

/**
 * @author: ionel.ortelecan
 */
public class ConnectivityTest extends AbstractBaseTest {

    public ConnectivityTest() {
        super();
    }


    @Before
    public void setUp() throws Exception {
        super.setUp();
    }

    @After
    public void tearDown() throws Exception {
        super.tearDown();
    }


    /**
     * Navigates to the "Large Scale Connectivity" page once.
     *
     * @throws Exception if something goes wrong
     */
    public void testPostConnectivityDataOnce() throws Exception {
        loginAdmin();
        postConnectivityData();
        logout();
    }


    /**
     * Navigates to the "Large Scale Connectivity" page ten times.
     *
     * @throws Exception if something goes wrong
     */
    public void testPostConnectivityDataTenTimes() throws Exception {
        loginAdmin();

        for (int i = 0; i < 10; i++) {
            postConnectivityData();
            webDriver.get(baseUrl + USER_PROFILE_URL_SUFFIX);
            WebElement logout = findElement(By.name("logout"));
            assertEquals("button", logout.getTagName());
            assertEquals("submit", logout.getAttribute("type"));
        }

        logout();
    }


    /**
     * Loads to the "Large Scale Connectivity" page.
     * <p/>
     * The posted data is: the default connectivity and the first surface found.
     *
     * @throws Exception if something goes wrong
     */
    private void postConnectivityData() throws Exception {
        WebElement connectivityLink = webDriver.findElement(By.id("nav-connectivity")).findElement(By.linkText("Connectivity"));
        connectivityLink.click();

        if (isElementPresent(By.id("nav-connectivity-connectivity"))) {
            WebElement largeConnectivityLink = webDriver.findElement(By.partialLinkText("Large Scale Connectivity"));
            largeConnectivityLink.click();
        } else {
            //select the default project first

        }
        WebElement header = findElement(By.tagName("hgroup")).findElement(By.tagName("h2"));
        assertNotNull("Couldn't find the header element.", header);
        assertEquals("The header is not correct.", "Fill parameters for step view - ConnectivityViewer", header.getText());

        WebElement select = findElement(By.id("surface_data"));
        List<WebElement> allOptions = select.findElements(By.tagName("option"));
        for (WebElement option : allOptions) {
            String value = option.getAttribute("value");
            if (value == null || value.trim().length() == 0) {
                continue;
            }
            option.click();
            break;
        }

        List<WebElement> allButtons = webDriver.findElements(By.tagName("button"));
        for (WebElement button : allButtons) {
            if ("Launch".equals(button.getText())) {
                button.click();
                break;
            }
        }

        //wait for surface to load
        Thread.sleep(5 * 1000);
        findElement(By.partialLinkText("3D Edges")).click();

        header = findElement(By.tagName("hgroup")).findElement(By.tagName("h2"));
        assertNotNull("Couldn't find the header element in connectivity page.", header);
        assertEquals("The header is not correct.", "Connectivity Control", header.getText());
    }
}
