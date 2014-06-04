/**
 * (c)  Baycrest Centre for Geriatric Care ("Baycrest"), 2012, all rights reserved.
 *
 * No redistribution or commercial re-sale is permitted.
 * Neither the name of Baycrest nor the names of its contributors may be used to endorse or promote
 * products or services derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY BAYCREST ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL BAYCREST BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 **/

package org.tvb.tests;

import com.google.common.base.Function;
import junit.framework.TestCase;
import org.junit.After;
import org.junit.Before;
import org.openqa.selenium.*;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxProfile;
import org.openqa.selenium.htmlunit.HtmlUnitDriver;
import org.openqa.selenium.support.ui.ExpectedCondition;
import org.openqa.selenium.support.ui.FluentWait;
import org.openqa.selenium.support.ui.Wait;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;


/**
 * @author: ionel.ortelecan
 */
public abstract class AbstractBaseTest extends TestCase {
    public String LOGIN_URL_SUFFIX = "/user/";
    public String USER_PROFILE_URL_SUFFIX = "/user/profile";

    public String ADMIN_USERNAME = "admin";
    public String ADMIN_PASSWORD = "pass";

    protected WebDriver webDriver;
    protected Wait<WebDriver> wait;
    protected String baseUrl = "http://127.0.0.1:8080";

    //the maximum period of time which we should wait for a burst to complete a certain operation (specified in seconds)
    protected Integer maxTimeToWaitForBurstToStart = 600;
    protected Integer maxTimeToWaitForBurstToFinish = 3600;
    protected Integer timeToWaitForBurstToLoad = 5;
    //how much we should wait for an element to appear on the page before make the test to fail (specified in seconds)
    protected Integer timeout = 30;

    private Logger logger = Logger.getLogger(this.getClass().getName());


    public AbstractBaseTest() {
        Properties props = new Properties();
        InputStream inputStream = this.getClass().getResourceAsStream("/config.properties");
        try {
            props.load(inputStream);
            baseUrl = readProperty(props, "baseUrl", baseUrl);
            maxTimeToWaitForBurstToStart = readIntegerProperty(props,
                    "maxTimeToWaitForBurstToStart", maxTimeToWaitForBurstToStart.toString());
            maxTimeToWaitForBurstToFinish = readIntegerProperty(props,
                    "maxTimeToWaitForBurstToFinish", maxTimeToWaitForBurstToFinish.toString());
            timeToWaitForBurstToLoad = readIntegerProperty(props,
                    "timeToWaitForBurstToLoad", timeToWaitForBurstToLoad.toString());
            timeout = readIntegerProperty(props, "timeout", timeout.toString());
        } catch (IOException e) {
            logger.warning(e.getMessage());
            logger.warning("The configuration file couldn't be loaded. " +
                    "All the properties defined there will use their default values.");
        }
    }


    @Before
    public void setUp() throws Exception {
//        FirefoxProfile profile = new FirefoxProfile();
//        profile.setPreference("webgl.force-enabled", true);
//        profile.setPreference("webgl.verbose", true);
//        webDriver = new FirefoxDriver(profile);

        webDriver = new HtmlUnitDriver(true);
        wait = new FluentWait<WebDriver>(webDriver)
                .withTimeout(timeout, TimeUnit.SECONDS)
                .pollingEvery(2, TimeUnit.SECONDS)
                .ignoring(NoSuchElementException.class);
    }


    @After
    public void tearDown() throws Exception {
        webDriver.quit();
    }


    /**
     * Search for an element using the given location mechanism.
     * It waits 30 seconds for the element to be present on the page,
     * checking for its presence once every 5 seconds.
     *
     * @param by The locating mechanism
     * @return The first matching element on the current page
     * @throws org.openqa.selenium.TimeoutException If the rime expires and no matching elements are found
     */
    protected WebElement findElement(final By by) throws TimeoutException {
        return wait.until(new Function<WebDriver, WebElement>() {
            public WebElement apply(WebDriver driver) {
                return driver.findElement(by);
            }
        });
    }


    /**
     * Finds an element by a js script. The script should return a dom element.
     *
     * @param scriptSelector the script that should be executed
     * @return the found element
     */
    protected WebElement findElementByScript(final String scriptSelector) {
        return wait.until(new Function<WebDriver, WebElement>() {
            public WebElement apply(WebDriver driver) {
                return (WebElement)((JavascriptExecutor)webDriver).executeScript(scriptSelector);
            }
        });
    }


    /**
     * Note: Call this method only when you want to make sure that a certain element was removed.
     * If you only want to check if the element is present use the method 'isElementPresent'.
     *
     * Check if an element was removed. This method waits for the element to be removed.
     * (waits until the set timeout period expires).

     * @param by The locating mechanism
     * @return <code>true</code> only if the asked element was removed. <code>false</code> only if the
     * timeout period expires and the element is still present.
     */
    protected boolean wasElementRemoved(final By by) {
        try {
            wait.until(new ExpectedCondition<Boolean>() {
                public Boolean apply(WebDriver d) {
                    return !isElementPresent(by);
                }
            });

            return true;
        } catch (TimeoutException e) {
            return false;
        }
    }


    /**
     * Used for checking if a certain element exists on the current page. The
     * method doesn't wait for the page to load.
     *
     * @param by The locating mechanism
     * @return <code>true</code> if the element is found on the current page
     */
    protected boolean isElementPresent(By by) {
        try {
            webDriver.findElement(by);
            return true;
        } catch (NoSuchElementException e) {
            return false;
        }
    }


    /**
     * Logs an administrator into the application.
     */
    protected void loginAdmin() {
        login(ADMIN_USERNAME, ADMIN_PASSWORD);
    }


    /**
     * Used for login into the application a certain user.
     *
     * @param usernameStr the username
     * @param passwordStr the password
     */
    protected void login(String usernameStr, String passwordStr) {
        webDriver.get(baseUrl + LOGIN_URL_SUFFIX);

        WebElement username = findElement(By.id("username"));
        WebElement password = findElement(By.id("password"));
        username.sendKeys(usernameStr);
        password.sendKeys(passwordStr);
        password.submit();
    }


    /**
     * Used for login out from the application a certain user.
     */
    protected void logout() {
        webDriver.get(baseUrl + USER_PROFILE_URL_SUFFIX);

        WebElement logout = findElement(By.name("logout"));
        assertEquals("button", logout.getTagName());
        assertEquals("submit", logout.getAttribute("type"));
        logout.submit();
    }


    /**
     * Returns the ask property from the given property file. If the obtained value
     * of the property is null or empty than the default value will be returned.
     *
     * @param props        a loaded property file
     * @param propertyKey  the key for the asked property
     * @param defaultValue the value which will be returned if the property is not defined
     * @return the value of the asked property
     */
    private String readProperty(Properties props, String propertyKey, String defaultValue) {
        String propertyValue = props.getProperty(propertyKey);
        if (propertyValue != null && propertyValue.trim().length() > 0) {
            return propertyValue;
        } else {
            logger.warning("The property " + propertyKey + " couldn't be obtained. " +
                    "The tests will use the default value for this property.");
            return defaultValue;
        }
    }

    private Integer readIntegerProperty(Properties props, String propertyKey, String defaultValue) {
        String propertyValue = readProperty(props, propertyKey, defaultValue);
        return Integer.valueOf(propertyValue);
    }
}
